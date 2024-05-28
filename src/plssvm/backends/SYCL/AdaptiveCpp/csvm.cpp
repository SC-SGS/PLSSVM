/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/AdaptiveCpp/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                 // plssvm::backend_type
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/device_ptr.hpp"                   // plssvm::adaptivecpp::detail::::device_ptr
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/queue_impl.hpp"                   // plssvm::adaptivecpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/AdaptiveCpp/detail/utility.hpp"                      // plssvm::adaptivecpp::detail::{get_device_list, device_synchronize, get_adaptivecpp_version_short, get_adaptivecpp_version}
#include "plssvm/backends/SYCL/exceptions.hpp"                                      // plssvm::adaptivecpp::backend_exception
#include "plssvm/backends/SYCL/implementation_types.hpp"                            // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel/cg_explicit/blas.hpp"                         // plssvm::sycl::detail::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/SYCL/kernel/cg_explicit/kernel_matrix_assembly.hpp"       // plssvm::sycl::detail::device_kernel_assembly
#include "plssvm/backends/SYCL/kernel/cg_implicit/kernel_matrix_assembly_blas.hpp"  // plssvm::sycl::detail::device_kernel_assembly_symm
#include "plssvm/backends/SYCL/kernel/predict_kernel.hpp"                           // plssvm::sycl::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/backends/SYCL/kernel_invocation_types.hpp"                         // plssvm::kernel_invocation_type
#include "plssvm/constants.hpp"                                                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                                                 // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                                                // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                            // plssvm::detail::memory_size
#include "plssvm/detail/performance_tracker.hpp"                                    // plssvm::detail::tracking_entry
#include "plssvm/detail/utility.hpp"                                                // plssvm::detail::get_system_memory
#include "plssvm/exceptions/exceptions.hpp"                                         // plssvm::exception
#include "plssvm/gamma.hpp"                                                         // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"                                         // plssvm::kernel_type
#include "plssvm/parameter.hpp"                                                     // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/shape.hpp"                                                         // plssvm::shape
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                              // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // ::sycl::range, ::sycl::nd_range, ::sycl::handler, ::sycl::info::device

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstddef>    // std::size_t
#include <cstdint>    // std::int32_t, std::uint16_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <limits>     // std::numeric_limits::max
#include <string>     // std::string
#include <tuple>      // std::tie
#include <variant>    // std::get
#include <vector>     // std::vector

namespace plssvm::adaptivecpp {

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

    // get all available devices wrt the requested target platform
    std::tie(devices_, target_) = detail::get_device_list(target);

    // set correct kernel invocation type if "automatic" has been provided
    if (invocation_type_ == sycl::kernel_invocation_type::automatic) {
        // always use nd_range for AdaptiveCpp
        invocation_type_ = sycl::kernel_invocation_type::nd_range;
        if (target_ == target_platform::cpu) {
#if !defined(__HIPSYCL_USE_ACCELERATED_CPU__) && defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
            plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                                "WARNING: the AdaptiveCpp automatic target for the CPU is set to nd_range, but AdaptiveCpp hasn't been build with the \"omp.accelerated\" compilation flow resulting in major performance losses!\n");
#endif
        }
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing AdaptiveCpp ({}) as SYCL backend with the kernel invocation type \"{}\" for the svm_kernel.\n",
                        detail::get_adaptivecpp_version_short(),
                        plssvm::detail::tracking_entry{ "backend", "sycl_kernel_invocation_type", invocation_type_ });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "dependencies", "adaptivecpp_version", detail::get_adaptivecpp_version() }));
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::sycl }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "sycl_implementation_type", plssvm::sycl::implementation_type::adaptivecpp }));

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", target_) };
    }

    // print found SYCL devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} SYCL device(s) for the target platform {}:\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() },
                        plssvm::detail::tracking_entry{ "backend", "target_platform", target_ });
    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::string device_name = devices_[device].impl->sycl_queue.get_device().template get_info<::sycl::info::device::name>();
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}]\n",
                            device,
                            device_name);
        device_names.emplace_back(device_name);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (const queue_type &q : devices_) {
            device_synchronize(q);
        }
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        const ::plssvm::detail::memory_size adaptivecpp_global_mem_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::global_mem_size>()) };
        if (target_ == target_platform::cpu) {
            plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                                "WARNING: the returned 'global_mem_size' for AdaptiveCpp targeting the CPU device {} is nonsensical ('std::numeric_limits<std::size_t>::max()'). Using 'get_system_memory()' instead.\n",
                                device_id);
            res[device_id] = std::min(adaptivecpp_global_mem_size, ::plssvm::detail::get_system_memory());
        } else {
            res[device_id] = adaptivecpp_global_mem_size;
        }
    }
    return res;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_mem_alloc_size>()) };
    }
    return res;
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    return devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::max_work_group_size>();
}

::plssvm::detail::dim_type csvm::get_max_grid_size([[maybe_unused]] const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);

    // fallback to maximum theoretical value, may break at runtime!
    ::sycl::id<3> native_range{};
    const std::size_t max_int32 = std::numeric_limits<std::int32_t>::max();
    const std::size_t max_uint16 = std::numeric_limits<std::uint16_t>::max();
    if (target_ == target_platform::cpu) {
        native_range = ::sycl::id<3>{ max_int32, max_int32, max_int32 };
    } else {
        native_range = ::sycl::id<3>{ max_int32, max_uint16, max_uint16 };
    }

    // note: account for SYCL's different iteration range!
    return { native_range[2], native_range[1], native_range[0] };
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const std::size_t num_rows_reduced = data_d.shape().x - 1;
    const std::size_t num_features = data_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(native_exec, sycl::detail::device_kernel_assembly<kernel_function_type::linear>{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y });
                });
                break;
            case kernel_function_type::polynomial:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::rbf:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::rbf, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::sigmoid:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::laplacian:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::laplacian, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::chi_squared:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::chi_squared, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
        }
    }
    detail::device_synchronize(device);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const ::plssvm::detail::execution_range &mirror_exec, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const std::size_t num_rhs = B_d.shape().x;
    const std::size_t num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (std::size_t i = 0; i < exec.grids.size(); ++i) {
        {
            // capturing structured bindings is a C++20 extension
            ::plssvm::detail::dim_type partial_grid{};
            ::plssvm::detail::dim_type offsets{};
            std::tie(partial_grid, offsets) = exec.grids[i];
            // convert execution range grid[i] to SYCL's native range<3>
            const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

            const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(native_exec, sycl::detail::device_kernel_symm{ cgh, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y });
            });
        }

        {
            const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

            if (num_mirror_rows > 0) {
                // capturing structured bindings is a C++20 extension
                ::plssvm::detail::dim_type partial_grid{};
                ::plssvm::detail::dim_type offsets{};
                std::tie(partial_grid, offsets) = mirror_exec.grids[i];
                // convert execution range grid[i] to SYCL's native range<3>
                const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

                const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(native_exec, sycl::detail::device_kernel_symm_mirror{ cgh, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y });
                });
            }
        }
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        device.impl->sycl_queue.parallel_for(native_exec, sycl::detail::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get(), offsets.x, offsets.y });
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        device.impl->sycl_queue.parallel_for(native_exec, sycl::detail::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale, offsets.x, offsets.y });
    }
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const std::size_t num_rows_reduced = A_d.shape().x - 1;
    const std::size_t num_features = A_d.shape().y;
    const std::size_t num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(native_exec, sycl::detail::device_kernel_assembly_symm<kernel_function_type::linear>{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y });
                });
                break;
            case kernel_function_type::polynomial:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::rbf:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::rbf, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::sigmoid:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::laplacian:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::laplacian, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::chi_squared:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::chi_squared, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
        }
    }
    detail::device_synchronize(device);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_sv = alpha_d.shape().y;
    const std::size_t device_specific_num_sv = sv_d.shape().x;
    const std::size_t num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const std::size_t sv_offset = data_distribution_->place_row_offset(device_id);

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(native_exec, sycl::detail::device_kernel_w_linear{ cgh, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.x, offsets.y });
        });
    }
    detail::device_synchronize(device);

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const std::size_t num_features = predict_points_d.shape().y;
    const std::size_t num_sv = sv_or_w_d.shape().x;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // convert execution range block to SYCL's native range<3>
    const ::sycl::range native_block = detail::dim_type_to_native<2>(exec.block);

    for (const auto &grid : exec.grids) {
        // capturing structured bindings is a C++20 extension
        ::plssvm::detail::dim_type partial_grid{};
        ::plssvm::detail::dim_type offsets{};
        std::tie(partial_grid, offsets) = grid;
        // convert execution range grid[i] to SYCL's native range<3>
        const ::sycl::range native_partial_grid = detail::dim_type_to_native<2>(partial_grid) * native_block;

        const ::sycl::nd_range native_exec{ native_partial_grid, native_block };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    cgh.parallel_for(native_exec, sycl::detail::device_kernel_predict_linear{ cgh, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.x, offsets.y });
                });
                break;
            case kernel_function_type::polynomial:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::rbf:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::rbf, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::sigmoid:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::laplacian:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::laplacian, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::chi_squared:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::chi_squared, real_type>;
                    cgh.parallel_for(native_exec, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma) });
                });
                break;
        }
    }
    detail::device_synchronize(device);

    return out_d;
}

}  // namespace plssvm::adaptivecpp
