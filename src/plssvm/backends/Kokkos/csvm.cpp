/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/csvm.hpp"

#include "plssvm/backends/execution_range.hpp"                                        // plssvm::detail::{execution_range, dim_type}
#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"                    // PLSSVM_KOKKOS_BACKEND_INVOKE_RETURN_IF_*, PLSSVM_KOKKOS_BACKEND_INVOKE_IF_
#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"                               // plssvm::kokkos::detail::device_ptr
#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"                           // plssvm::kokkos::detail::{device_wrapper, get_device_list}
#include "plssvm/backends/Kokkos/detail/utility.hpp"                                  // plssvm::kokkos::detail::{available_target_platform_to_execution_space_mapping, get_kokkos_version, dim_type_to_native, get_device_name, device_synchronize}
#include "plssvm/backends/Kokkos/exceptions.hpp"                                      // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"                                 // plssvm::kokkos::{execution_space, list_available_execution_spaces}
#include "plssvm/backends/Kokkos/kernel/cg_explicit/blas.hpp"                         // plssvm::kokkos::detail::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/Kokkos/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::kokkos::detail::device_kernel_assembly
#include "plssvm/backends/Kokkos/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::kokkos::detail::device_kernel_assembly_symm
#include "plssvm/backends/Kokkos/kernel/predict_kernel.hpp"                           // plssvm::kokkos::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                       // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE, plssvm::FEATURE_BLOCK_SIZE
#include "plssvm/detail/assert.hpp"                                                   // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"                                        // plssvm::detail::triangular_data_distribution
#include "plssvm/detail/logging.hpp"                                                  // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                              // plssvm::detail::memory_size
#include "plssvm/detail/tracking/performance_tracker.hpp"                             // plssvm::detail::tracking::tracking_entry
#include "plssvm/detail/type_traits.hpp"                                              // plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"                                                  // plssvm::detail::{get_system_memory, unreachable}
#include "plssvm/exceptions/exceptions.hpp"                                           // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                // plssvm::verbosity_level

#include "Kokkos_Core.hpp"  // Kokkos::TeamPolicy, Kokkos::ParallelForTag, Kokkos::parallel_for, Kokkos::PerTeam
                            // Kokkos::Experimental::HPX::impl_max_hardware_threads, Kokkos::OpenMP::impl_max_hardware_threads, Kokkos::Threads::impl_max_hardware_threads

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::format

#include <cmath>      // std::sqrt
#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <limits>     // std::numeric_limits::max
#include <map>        // std::map
#include <string>     // std::string
#include <utility>    // std::move
#include <vector>     // std::vector

// a dummy class used as functor to the team_size_max function
template <typename ExecutionSpace>
struct dummy {
    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<ExecutionSpace>::member_type &) const { }
};

namespace plssvm::kokkos {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check whether the requested target platform has been enabled
    switch (target) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
            break;
    }

    // check whether the requested execution space is available
    if (!::plssvm::detail::contains(list_available_execution_spaces(), space_)) {
        throw backend_exception{ fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space_, fmt::join(list_available_execution_spaces(), ", ")) };
    }

    // get all available target_platform <-> Kokkos::ExecutionSpace combinations
    const std::map<target_platform, std::vector<execution_space>> available_combinations = detail::available_target_platform_to_execution_space_mapping();

    // check whether the provided execution space is the automatic one
    if (space_ == execution_space::automatic) {
        // automatically determine the execution space and potentially automatically determine the target platform
        if (target == target_platform::automatic) {
            // go through all combinations and choose the first execution space in order: gpu_nvidia -> gpu_amd -> gpu_intel -> cpu
            for (const target_platform target_order : list_available_target_platforms()) {
                if (::plssvm::detail::contains(available_combinations, target_order)) {
                    // the target platform is supported -> choose the first execution space to use in the Kokkos backend
                    space_ = available_combinations.at(target_order).front();
                    target_ = target_order;
                    break;
                }
            }
        } else {
            // check whether the provided target platform is compatible with the currently available Kokkos::ExecutionSpaces
            if (::plssvm::detail::contains(available_combinations, target)) {
                // the target platform is supported -> choose the first execution space to use in the Kokkos backend
                space_ = available_combinations.at(target).front();
                target_ = target;
            } else {
                // the provided target platform is unsupported -> throw an exception
                throw backend_exception{ fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform {}!", fmt::join(list_available_execution_spaces(), ", "), target) };
            }
        }

        // output what we use as automatic Kokkos execution space
        plssvm::detail::log(verbosity_level::full,
                            "\nUsing {} as automatic Kokkos::ExecutionSpace.",
                            space_);
    } else {
        // execution space explicitly provided and potentially automatically determine the target platform
        if (target == target_platform::automatic) {
            // go through all combinations (gpu_nvidia -> gpu_amd -> gpu_intel -> cpu) and check whether the requested execution space supports that target platform
            for (const target_platform target_order : list_available_target_platforms()) {
                if (::plssvm::detail::contains(available_combinations, target_order) && ::plssvm::detail::contains(available_combinations.at(target_order), space_)) {
                    // the provided execution space supports the target platform
                    target_ = target_order;
                    break;
                }
            }
        } else {
            if (!::plssvm::detail::contains(available_combinations, target) || !::plssvm::detail::contains(available_combinations.at(target), space_)) {
                // the provided execution space and target platform combination is unsupported
                throw backend_exception{ fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform {}!", space_, target) };
            }
        }
    }

    // At this point, space_ may NEVER be execution_space::automatic!
    PLSSVM_ASSERT(space_ != execution_space::automatic, "At this point, the Kokkos execution space must be determined and must NOT be automatic!");

    // Kokkos::Experimental::OpenMPTarget and Kokkos::Experimental::OpenACC currently not supported!
    if (space_ == execution_space::openmp_target || space_ == execution_space::openacc) {
        throw backend_exception{ fmt::format("The Kokkos execution space {} is currently not supported!", space_) };
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing Kokkos ({}) as backend with the Kokkos::ExecutionSpace {}.\n",
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_version", detail::get_kokkos_version() },
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_default_execution_space", space_ });

    // output automatic target platform information
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    }

    // get all available devices wrt the requested target platform
    devices_ = detail::get_device_list(space_, target_);

    // throw exception if no devices in the current execution space could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("No devices found for the Kokkos execution space {} with the target platform {}!", space_, target_) };
    }

    // print found Kokkos devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} Kokkos device(s) for the target platform {}:\n",
                        plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ });

    std::vector<std::string> device_names{};
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = detail::get_device_name(devices_[device]);
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n",
                            device,
                            device_name);
        device_names.emplace_back(device_name);
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the CUDA devices have finished before destruction
        for (const queue_type &device : devices_) {
            detail::device_synchronize(device);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    PLSSVM_ASSERT(space_ != execution_space::automatic, "The automatic execution_space may not be provided to this function!");

    std::vector<::plssvm::detail::memory_size> device_memory(this->num_available_devices());
    switch (space_) {
        case execution_space::automatic:
            throw backend_exception{ "Unsupported execution_space::automatic provided!" };
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    device_memory[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].get<execution_space::cuda>().cuda_device_prop().totalGlobalMem) };
                }
            });
            break;
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    device_memory[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].get<execution_space::hip>().hip_device_prop().totalGlobalMem) };
                }
            });
            break;
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    device_memory[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].get<execution_space::sycl>().sycl_queue().get_device().get_info<::sycl::info::device::global_mem_size>()) };
                }
            });
            break;
        case execution_space::hpx:
        case execution_space::openmp:
        case execution_space::threads:
        case execution_space::serial:
            // NOTE: for these execution spaces, this->num_available_devices will always return 1
            PLSSVM_ASSERT(this->num_available_devices() == 1, "The host side Kokkos execution spaces should always only be represented using a single device!");
            device_memory[0] = ::plssvm::detail::get_system_memory();
            break;
        // TODO: implement for Kokkos::Experimental::OpenMPTarget and Kokkos::Experimental::OpenACC
        case execution_space::openmp_target:
        case execution_space::openacc:
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    return device_memory;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    PLSSVM_ASSERT(space_ != execution_space::automatic, "The automatic execution_space may not be provided to this function!");

    std::vector<::plssvm::detail::memory_size> max_mem_alloc_size(this->num_available_devices());
    switch (space_) {
        case execution_space::automatic:
            throw backend_exception{ "Unsupported execution_space::automatic provided!" };
        case execution_space::cuda:
        case execution_space::hip:
            max_mem_alloc_size = this->get_device_memory();
            break;
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    max_mem_alloc_size[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].get<execution_space::sycl>().sycl_queue().get_device().get_info<::sycl::info::device::max_mem_alloc_size>()) };
                }
            });
            break;
        case execution_space::hpx:
        case execution_space::openmp:
        case execution_space::threads:
        case execution_space::serial:
            max_mem_alloc_size = this->get_device_memory();
            break;
        // TODO: implement for Kokkos::Experimental::OpenMPTarget and Kokkos::Experimental::OpenACC
        case execution_space::openmp_target:
        case execution_space::openacc:
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    return max_mem_alloc_size;
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    PLSSVM_ASSERT(space_ != execution_space::automatic, "The automatic execution_space may not be provided to this function!");

    // NOTE: the maximum theoretical work-group size, may be additionally limited by the amount of used scratch memory
    return devices_[device_id].execute_and_return([](const auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        // NOTE: CUDA + HIP + SYCL: returns the maximum possible number of threads, due to no further limitations in the dummy functor (like, e.g., scratch memory)
        // NOTE: HPX + Serial: hardcoded to 1
        // NOTE: OpenMP: should be 1-2; most likely 1
        // NOTE: Threads: should be equal to number of hardware threads IF hwloc is enabled; otherwise 1
        // NOTE: OpenMPTarget: hardcoded to 256
        // NOTE: OpenACC: hardcoded to 512

        // NOTE: the functor types doesn't matter -> the dummy class
        return Kokkos::TeamPolicy<kokkos_execution_space_type>{}.team_size_max(dummy<kokkos_execution_space_type>{}, Kokkos::ParallelForTag{});
    });
}

::plssvm::detail::dim_type csvm::get_max_grid_size([[maybe_unused]] const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    PLSSVM_ASSERT(space_ != execution_space::automatic, "The automatic execution_space may not be provided to this function!");

    // NOTE: Kokkos only supports one-dimensional execution ranges!
    // NOTE: we only use two-dimensional kernels!
    switch (space_) {
        case execution_space::automatic:
            throw backend_exception{ "Unsupported execution_space::automatic provided!" };
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_RETURN_IF_CUDA(([&]() -> ::plssvm::detail::dim_type {
                const cudaDeviceProp &prop = devices_[device_id].get<execution_space::cuda>().cuda_device_prop();
                const auto max_grid_size = static_cast<unsigned long long>(std::sqrt(prop.maxGridSize[0]));
                return { max_grid_size, max_grid_size, 1ull };
            }));
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_RETURN_IF_HIP(([&]() -> ::plssvm::detail::dim_type {
                const hipDeviceProp_t &prop = devices_[device_id].get<execution_space::hip>().hip_device_prop();
                const auto max_grid_size = static_cast<unsigned long long>(std::sqrt(prop.maxGridSize[0]));
                return { max_grid_size, max_grid_size, 1ull };
            }));
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_RETURN_IF_SYCL(([&]() -> ::plssvm::detail::dim_type {
            // TODO: replace with standardized function if there will be one in the future
#if defined(SYCL_EXT_ONEAPI_MAX_WORK_GROUP_QUERY)
                const ::sycl::id<3> native_range = devices_[device_id].get<execution_space::sycl>().sycl_queue().get_device().get_info<::sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
#else
                // fallback to maximum theoretical value, may break at runtime!
                ::sycl::id<3> native_range{};
                const std::size_t max_int32 = std::numeric_limits<std::int32_t>::max();
                const std::size_t max_uint16 = std::numeric_limits<std::uint16_t>::max();
                if (target_ == target_platform::cpu) {
                    native_range = ::sycl::id<3>{ max_int32, max_int32, max_int32 };
                } else {
                    native_range = ::sycl::id<3>{ max_int32, max_uint16, max_uint16 };
                }
#endif
                // note: account for SYCL's different iteration range!
                return { native_range[2], native_range[1], native_range[0] };
            }));
        case execution_space::hpx:
        case execution_space::openmp:
        case execution_space::threads:
        case execution_space::serial:
            return { std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1ull };
        case execution_space::openmp_target:
        case execution_space::openacc:
            // TODO: implement for Kokkos::Experimental::OpenMPTarget and Kokkos::Experimental::OpenACC
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.shape().x - 1;
    const unsigned long long num_features = data_d.shape().y;

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, devices_[device_id] };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;
    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team size
    const int team_size = detail::dim_type_to_native(exec.block);

    return devices_[device_id].execute_and_return([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            switch (params.kernel_type) {
                case kernel_function_type::linear:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::linear>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x });
                    }
                    break;
                case kernel_function_type::polynomial:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::rbf:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::rbf, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::sigmoid:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::laplacian:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::laplacian, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::chi_squared:
                    {
                        using functor_type = detail::device_kernel_assembly<kokkos_execution_space_type, kernel_function_type::chi_squared, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_explicit_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get().get<space>(), data_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get().get<space>(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
            }
        }
        detail::device_synchronize(device);

        return std::move(kernel_matrix_d);
    });
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const unsigned long long num_rhs = B_d.shape().x;
    const unsigned long long num_rows = B_d.shape().y;

    devices_[device_id].execute([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        // calculate the number of data points this device is responsible for
        const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
        // get the offset of the data points this device is responsible for
        const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);
        // the necessary amount of scratch memory for the kernels
        const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

        // save the team size
        const int team_size = detail::dim_type_to_native(exec.block);

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            Kokkos::parallel_for("blas_level_3_kernel_explicit", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_symm<kokkos_execution_space_type>{ num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get().get<space>(), B_d.get().get<space>(), beta, C_d.get().get<space>(), offsets.x, offsets.y, partial_grid.x });
        }

        // save the team size
        const int mirror_team_size = detail::dim_type_to_native(mirror_exec.block);

        for (const auto &[partial_grid, offsets] : mirror_exec.grids) {
            const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

            if (num_mirror_rows > 0) {
                // convert execution range partial_grid to Kokkos' native one-dimensional size
                const int native_partial_grid = detail::dim_type_to_native(partial_grid);

                // create a Kokkos TeamPolicy
                Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, mirror_team_size };

                Kokkos::parallel_for("blas_level_3_kernel_explicit_mirror", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_symm_mirror<kokkos_execution_space_type>{ num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get().get<space>(), B_d.get().get<space>(), beta, C_d.get().get<space>(), offsets.x, offsets.y, partial_grid.x });
            }
        }
        detail::device_synchronize(device);
    });
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const unsigned long long num_rhs = lhs_d.shape().x;

    devices_[device_id].execute([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        // save the team size
        const int team_size = detail::dim_type_to_native(exec.block);

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            const Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            Kokkos::parallel_for("inplace_matrix_addition", team_policy, detail::device_kernel_inplace_matrix_add<kokkos_execution_space_type>{ num_rhs, lhs_d.get().get<space>(), rhs_d.get().get<space>(), offsets.x, offsets.y, partial_grid.x });
        }
        detail::device_synchronize(device);
    });
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const unsigned long long num_rhs = lhs_d.shape().x;

    devices_[device_id].execute([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        // save the team size
        const int team_size = detail::dim_type_to_native(exec.block);

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            const Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            Kokkos::parallel_for("inplace_matrix_scale", team_policy, detail::device_kernel_inplace_matrix_scale<kokkos_execution_space_type>{ num_rhs, lhs_d.get().get<space>(), scale, offsets.x, offsets.y, partial_grid.x });
        }
        detail::device_synchronize(device);
    });
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const unsigned long long num_rows_reduced = A_d.shape().x - 1;
    const unsigned long long num_features = A_d.shape().y;
    const unsigned long long num_classes = B_d.shape().x;

    devices_[device_id].execute([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        // calculate the number of data points this device is responsible for
        const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
        // get the offset of the data points this device is responsible for
        const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

        const real_type cost_factor = real_type{ 1.0 } / params.cost;
        const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

        // save the team size
        const int team_size = detail::dim_type_to_native(exec.block);

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            switch (params.kernel_type) {
                case kernel_function_type::linear:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::linear>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x });
                    }
                    break;
                case kernel_function_type::polynomial:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::rbf:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::rbf, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::sigmoid:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::laplacian:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::laplacian, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::chi_squared:
                    {
                        using functor_type = detail::device_kernel_assembly_symm<kokkos_execution_space_type, kernel_function_type::chi_squared, real_type>;
                        Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get().get<space>(), A_d.get().get<space>(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get().get<space>(), C_d.get().get<space>(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
            }
        }
        detail::device_synchronize(device);
    });
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = alpha_d.shape().y;
    const unsigned long long device_specific_num_sv = sv_d.shape().x;
    const unsigned long long num_features = sv_d.shape().y;

    // get the offset of the data points this device is responsible for
    const unsigned long long sv_offset = data_distribution_->place_row_offset(device_id);

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[device_id] };

    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team size
    const int team_size = detail::dim_type_to_native(exec.block);

    return devices_[device_id].execute_and_return([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            Kokkos::parallel_for("w_kernel", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_w_linear<kokkos_execution_space_type>{ w_d.get().get<space>(), alpha_d.get().get<space>(), sv_d.get().get<space>(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.x, offsets.y, partial_grid.x });
        }
        detail::device_synchronize(device);

        return std::move(w_d);
    });
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const unsigned long long num_features = predict_points_d.shape().y;
    const unsigned long long num_sv = sv_or_w_d.shape().x;

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[device_id] };

    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team size
    const int team_size = detail::dim_type_to_native(exec.block);

    return devices_[device_id].execute_and_return([&](auto &device) {
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(device)>;
        constexpr execution_space space = kokkos_type_to_execution_space_v<kokkos_execution_space_type>;

        for (const auto &[partial_grid, offsets] : exec.grids) {
            // convert execution range partial_grid to Kokkos' native one-dimensional size
            const int native_partial_grid = detail::dim_type_to_native(partial_grid);

            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<kokkos_execution_space_type> team_policy{ device, native_partial_grid, team_size };

            switch (params.kernel_type) {
                case kernel_function_type::linear:
                    {
                        using functor_type = detail::device_kernel_predict_linear<kokkos_execution_space_type>;
                        Kokkos::parallel_for("predict_kernel_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), sv_or_w_d.get().get<space>(), rho_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x });
                    }
                    break;
                case kernel_function_type::polynomial:
                    {
                        using functor_type = detail::device_kernel_predict<kokkos_execution_space_type, kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("predict_kernel_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), alpha_d.get().get<space>(), rho_d.get().get<space>(), sv_or_w_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::rbf:
                    {
                        using functor_type = detail::device_kernel_predict<kokkos_execution_space_type, kernel_function_type::rbf, real_type>;
                        Kokkos::parallel_for("predict_kernel_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), alpha_d.get().get<space>(), rho_d.get().get<space>(), sv_or_w_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::sigmoid:
                    {
                        using functor_type = detail::device_kernel_predict<kokkos_execution_space_type, kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                        Kokkos::parallel_for("predict_kernel_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), alpha_d.get().get<space>(), rho_d.get().get<space>(), sv_or_w_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                    }
                    break;
                case kernel_function_type::laplacian:
                    {
                        using functor_type = detail::device_kernel_predict<kokkos_execution_space_type, kernel_function_type::laplacian, real_type>;
                        Kokkos::parallel_for("predict_kernel_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), alpha_d.get().get<space>(), rho_d.get().get<space>(), sv_or_w_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
                case kernel_function_type::chi_squared:
                    {
                        using functor_type = detail::device_kernel_predict<kokkos_execution_space_type, kernel_function_type::chi_squared, real_type>;
                        Kokkos::parallel_for("predict_kernel_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get().get<space>(), alpha_d.get().get<space>(), rho_d.get().get<space>(), sv_or_w_d.get().get<space>(), predict_points_d.get().get<space>(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                    }
                    break;
            }
        }
        detail::device_synchronize(device);

        return std::move(out_d);
    });
}

}  // namespace plssvm::kokkos
