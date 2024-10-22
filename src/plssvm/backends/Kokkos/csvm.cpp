/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/csvm.hpp"

#include "plssvm/backends/execution_range.hpp"                                        // plssvm::detail::{execution_range, dim_type}
#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"                    // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"                               // plssvm::kokkos::detail::device_ptr
#include "plssvm/backends/Kokkos/detail/utility.hpp"                                  // plssvm::kokkos::detail::get_runtime_version
#include "plssvm/backends/Kokkos/exceptions.hpp"                                      // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"                                 // plssvm::kokkos::execution_space
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
#include "plssvm/detail/utility.hpp"                                                  // plssvm::detail::{get_system_memory, unreachable}
#include "plssvm/exceptions/exceptions.hpp"                                           // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                           // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                                       // plssvm::parameter
#include "plssvm/target_platforms.hpp"                                                // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                // plssvm::verbosity_level

#include "Kokkos_Core.hpp"  // TODO: docu

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::format

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <string>     // std::string
#include <vector>     // std::vector

namespace plssvm::kokkos {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(target_platform target, parameter params) :
    base_type{ params },
    space_{ determine_execution_space<Kokkos::DefaultExecutionSpace>() } {
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

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing Kokkos ({}) as backend with the Kokkos::DefaultExecutionSpace \"{}\".\n",
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_version", detail::get_kokkos_version() },
                        plssvm::detail::tracking::tracking_entry{ "dependencies", "kokkos_default_execution_space", space_ });

    // check whether the provided target platform is compatible with the Kokkos execution space
    if (target == target_platform::automatic) {
        // determine the default target based on the provided Kokkos execution space
        target_ = detail::determine_default_target_platform_from_execution_space(space_);
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    } else {
        // check whether the provided target platform is compatible with the execution space
        // throws a backend exception if the combination is invalid
        detail::check_execution_space_target_platform_combination(space_, target);
        target_ = target;
    }

    // get all available devices wrt the requested target platform
    devices_ = detail::get_device_list(space_, target_);

    // throw exception if no devices in the current execution space could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("Not devices found for the Kokkos execution space {} with the target platform {}!", space_, target_) };
    }

    // print found Kokkos devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} Kokkos device(s) for the target platform {}:\n",
                        plssvm::detail::tracking::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking::tracking_entry{ "backend", "target_platform", target_ });

    std::vector<std::string> device_names{};
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = detail::get_device_name(space_, devices_[device]);
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
    // TODO: implement for other execution spaces
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    switch (space_) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].cuda_device_prop().totalGlobalMem) };
                }
                return res;
            });
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].hip_device_prop().totalGlobalMem) };
                }
                return res;
            });
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].sycl_queue().get_device().get_info<::sycl::info::device::global_mem_size>()) };
                }
                return res;
            });
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            return std::vector<::plssvm::detail::memory_size>(this->num_available_devices(), ::plssvm::detail::get_system_memory());
        case execution_space::openmp_target:
        case execution_space::openacc:
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    // TODO: implement for other execution spaces
    switch (space_) {
        case execution_space::cuda:
        case execution_space::hip:
            return this->get_device_memory();
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                    res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].sycl_queue().get_device().get_info<::sycl::info::device::max_mem_alloc_size>()) };
                }
                return res;
            });
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            return this->get_device_memory();
        case execution_space::openmp_target:
        case execution_space::openacc:
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);

    // TODO: implement for other execution spaces
    switch (space_) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA([&]() {
                return static_cast<std::size_t>(devices_[device_id].cuda_device_prop().maxThreadsPerBlock);
            });
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP([&]() {
                return static_cast<std::size_t>(devices_[device_id].hip_device_prop().maxThreadsPerBlock);
            });
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL([&]() {
                return devices_[device_id].sycl_queue().get_device().get_info<::sycl::info::device::max_work_group_size>();
            });
        case execution_space::openmp_target:
        case execution_space::openacc:
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
            throw backend_exception{ fmt::format("Currently not implemented for the execution space: {}!", space_) };
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

::plssvm::detail::dim_type csvm::get_max_grid_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);

    // TODO: implement for other execution spaces
    switch (space_) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA(([&]() -> ::plssvm::detail::dim_type {
                const cudaDeviceProp &prop = devices_[device_id].cuda_device_prop();
                return { static_cast<std::size_t>(prop.maxGridSize[0]), static_cast<std::size_t>(prop.maxGridSize[1]), static_cast<std::size_t>(prop.maxGridSize[2]) };
            }));
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP(([&]() -> ::plssvm::detail::dim_type {
                const hipDeviceProp &prop = devices_[device_id].hip_device_prop();
                return { static_cast<std::size_t>(prop.maxGridSize[0]), static_cast<std::size_t>(prop.maxGridSize[1]), static_cast<std::size_t>(prop.maxGridSize[2]) };
            }));
        case execution_space::sycl:
        case execution_space::openmp_target:
        case execution_space::openacc:
        case execution_space::openmp:
        case execution_space::hpx:
        case execution_space::threads:
        case execution_space::serial:
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
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;
    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy(device, static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO);

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::linear>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x });
                }
                break;
            case kernel_function_type::polynomial:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::rbf:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::rbf, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::sigmoid:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::laplacian:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::laplacian, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::chi_squared:
                {
                    using functor_type = detail::device_kernel_assembly<kernel_function_type::chi_squared, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_explicit_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
        }
    }
    detail::device_synchronize(device);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const unsigned long long num_rhs = B_d.shape().x;
    const unsigned long long num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);
    // the necessary amount of scratch memory for the kernels
    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy{ device, static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO };

        Kokkos::parallel_for("blas_level_3_kernel_explicit", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_symm(num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y, partial_grid.x));
    }

    // save the mirror team sizes
    const ::plssvm::detail::dim_type mirror_team_sizes = mirror_exec.block;

    for (const auto &[partial_grid, offsets] : mirror_exec.grids) {
        const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

        if (num_mirror_rows > 0) {
            // create a Kokkos TeamPolicy
            Kokkos::TeamPolicy<> team_policy{ static_cast<int>(partial_grid.total_size()), static_cast<int>(mirror_team_sizes.total_size()), Kokkos::AUTO };

            Kokkos::parallel_for("blas_level_3_kernel_explicit_mirror", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_symm_mirror(num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y, partial_grid.x));
        }
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy{ static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO };

        Kokkos::parallel_for("inplace_matrix_addition", team_policy, detail::device_kernel_inplace_matrix_add(num_rhs, lhs_d.get(), rhs_d.get(), offsets.x, offsets.y, partial_grid.x));
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy{ static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO };

        Kokkos::parallel_for("inplace_matrix_scale", team_policy, detail::device_kernel_inplace_matrix_scale(num_rhs, lhs_d.get(), scale, offsets.x, offsets.y, partial_grid.x));
    }
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const unsigned long long num_rows_reduced = A_d.shape().x - 1;
    const unsigned long long num_features = A_d.shape().y;
    const unsigned long long num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    const real_type cost_factor = real_type{ 1.0 } / params.cost;
    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy(device, static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO);

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::linear>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x });
                }
                break;
            case kernel_function_type::polynomial:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::rbf:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::rbf, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::sigmoid:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::laplacian:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::laplacian, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::chi_squared:
                {
                    using functor_type = detail::device_kernel_assembly_symm<kernel_function_type::chi_squared, real_type>;
                    Kokkos::parallel_for("assemble_kernel_matrix_implicit_blas_level_3_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
        }
    }
    detail::device_synchronize(device);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = alpha_d.shape().y;
    const unsigned long long device_specific_num_sv = sv_d.shape().x;
    const unsigned long long num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const unsigned long long sv_offset = data_distribution_->place_row_offset(device_id);

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy{ static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO };

        Kokkos::parallel_for("w_kernel", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), detail::device_kernel_w_linear(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.x, offsets.y, partial_grid.x));
    }
    detail::device_synchronize(device);

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const unsigned long long num_features = predict_points_d.shape().y;
    const unsigned long long num_sv = sv_or_w_d.shape().x;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    const std::size_t scratch_memory_size = static_cast<std::size_t>(2u * FEATURE_BLOCK_SIZE * THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * sizeof(real_type);

    // save the team sizes
    const ::plssvm::detail::dim_type team_sizes = exec.block;

    for (const auto &[partial_grid, offsets] : exec.grids) {
        // create a Kokkos TeamPolicy
        Kokkos::TeamPolicy<> team_policy{ static_cast<int>(partial_grid.total_size()), static_cast<int>(team_sizes.total_size()), Kokkos::AUTO };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                {
                    using functor_type = detail::device_kernel_predict_linear;
                    Kokkos::parallel_for("predict_kernel_linear", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x });
                }
                break;
            case kernel_function_type::polynomial:
                {
                    using functor_type = detail::device_kernel_predict<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("predict_kernel_polynomial", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::rbf:
                {
                    using functor_type = detail::device_kernel_predict<kernel_function_type::rbf, real_type>;
                    Kokkos::parallel_for("predict_kernel_rbf", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::sigmoid:
                {
                    using functor_type = detail::device_kernel_predict<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    Kokkos::parallel_for("predict_kernel_sigmoid", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma), params.coef0 });
                }
                break;
            case kernel_function_type::laplacian:
                {
                    using functor_type = detail::device_kernel_predict<kernel_function_type::laplacian, real_type>;
                    Kokkos::parallel_for("predict_kernel_laplacian", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
            case kernel_function_type::chi_squared:
                {
                    using functor_type = detail::device_kernel_predict<kernel_function_type::chi_squared, real_type>;
                    Kokkos::parallel_for("predict_kernel_chi_squared", team_policy.set_scratch_size(0, Kokkos::PerTeam(scratch_memory_size)), functor_type{ out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, partial_grid.x, std::get<real_type>(params.gamma) });
                }
                break;
        }
    }
    detail::device_synchronize(device);

    return out_d;
}

}  // namespace plssvm::kokkos
