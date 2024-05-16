/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                 // plssvm::backend_type
#include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"                         // plssvm::dpcpp::detail::::device_ptr
#include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"                         // plssvm::dpcpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/DPCPP/detail/utility.hpp"                            // plssvm::dpcpp::detail::{get_device_list, device_synchronize, get_dpcpp_version}
#include "plssvm/backends/SYCL/exceptions.hpp"                                      // plssvm::dpcpp::backend_exception
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
#include "plssvm/exceptions/exceptions.hpp"                                         // plssvm::exception
#include "plssvm/gamma.hpp"                                                         // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"                                         // plssvm::kernel_type
#include "plssvm/parameter.hpp"                                                     // plssvm::parameter
#include "plssvm/shape.hpp"                                                         // plssvm::shape
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                              // plssvm::verbosity_level

#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler, sycl::info::device

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <string>     // std::string
#include <tuple>      // std::tie
#include <variant>    // std::get
#include <vector>     // std::vector

namespace plssvm::dpcpp {

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
        // always use nd_range for DPC++
        invocation_type_ = sycl::kernel_invocation_type::nd_range;
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing DPC++ ({}; {}) as SYCL backend with the kernel invocation type \"{}\" for the svm_kernel.\n",
                        plssvm::detail::tracking_entry{ "dependencies", "dpcpp_version", detail::get_dpcpp_version() },
                        plssvm::detail::tracking_entry{ "dependencies", "dpcpp_timestamp_version", detail::get_dpcpp_timestamp_version() },
                        plssvm::detail::tracking_entry{ "backend", "sycl_kernel_invocation_type", invocation_type_ });
    if (target == target_platform::automatic) {
        plssvm::detail::log(verbosity_level::full,
                            "Using {} as automatic target platform.\n",
                            target_);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::sycl }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "sycl_implementation_type", plssvm::sycl::implementation_type::dpcpp }));

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
    } catch (const plssvm::exception &e) {
        std::cout << e.what() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(devices_[device_id].impl->sycl_queue.get_device().get_info<::sycl::info::device::global_mem_size>()) };
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

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const std::size_t num_rows_reduced = data_d.shape().x - 1;
    const std::size_t num_features = data_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced - row_offset) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_);
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_assembly<kernel_function_type::linear>{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor });
            });
            break;
        case kernel_function_type::polynomial:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                cgh.parallel_for(execution_range, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree, std::get<real_type>(params.gamma), params.coef0 });
            });
            break;
        case kernel_function_type::rbf:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::rbf, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, std::get<real_type>(params.gamma) });
            });
            break;
        case kernel_function_type::sigmoid:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                cgh.parallel_for(execution_range, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, std::get<real_type>(params.gamma), params.coef0 });
            });
            break;
        case kernel_function_type::laplacian:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::laplacian, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, std::get<real_type>(params.gamma) });
            });
            break;
        case kernel_function_type::chi_squared:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly<kernel_function_type::chi_squared, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, std::get<real_type>(params.gamma) });
            });
            break;
    }
    detail::device_synchronize(device);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const std::size_t num_rhs = B_d.shape().x;
    const std::size_t num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }

    {
        const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
        const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                     static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
        const ::sycl::nd_range<2> execution_range{ grid, block };

        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(execution_range, sycl::detail::device_kernel_symm{ cgh, num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
        });
    }

    {
        const std::size_t num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

        if (num_mirror_rows > 0) {
            const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
            const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_mirror_rows) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                         static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
            const ::sycl::nd_range<2> execution_range{ grid, block };

            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_symm_mirror{ cgh, num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get() });
            });
        }
    }
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const std::size_t num_rows = lhs_d.shape().y;
    const queue_type &device = devices_[device_id];

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    device.impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_inplace_matrix_add{ num_rhs, lhs_d.get(), rhs_d.get() });
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, device_ptr_type &lhs_d, const real_type scale) const {
    const std::size_t num_rhs = lhs_d.shape().x;
    const std::size_t num_rows = lhs_d.shape().y;
    const queue_type &device = devices_[device_id];

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    device.impl->sycl_queue.parallel_for(execution_range, sycl::detail::device_kernel_inplace_matrix_scale{ num_rhs, lhs_d.get(), scale });
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const std::size_t num_rows_reduced = A_d.shape().x - 1;
    const std::size_t num_features = A_d.shape().y;
    const std::size_t num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_rows_reduced - row_offset) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, sycl::detail::device_kernel_assembly_symm<kernel_function_type::linear>{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes });
            });
            break;
        case kernel_function_type::polynomial:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                cgh.parallel_for(execution_range, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, params.degree, std::get<real_type>(params.gamma), params.coef0 });
            });
            break;
        case kernel_function_type::rbf:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::rbf, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, std::get<real_type>(params.gamma) });
            });
            break;
        case kernel_function_type::sigmoid:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                cgh.parallel_for(execution_range, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, std::get<real_type>(params.gamma), params.coef0 });
            });
            break;
        case kernel_function_type::laplacian:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::laplacian, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, std::get<real_type>(params.gamma) });
            });
            break;
        case kernel_function_type::chi_squared:
            device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                using functor_type = sycl::detail::device_kernel_assembly_symm<kernel_function_type::chi_squared, real_type>;
                cgh.parallel_for(execution_range, functor_type{ cgh, alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, std::get<real_type>(params.gamma) });
            });
            break;
    }
    detail::device_synchronize(device);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_sv = alpha_d.shape().y;
    const std::size_t device_specific_num_sv = sv_d.shape().x;
    const std::size_t num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const std::size_t sv_offset = data_distribution_->place_row_offset(device_id);

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                 static_cast<std::size_t>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
    const ::sycl::nd_range<2> execution_range{ grid, block };

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
        cgh.parallel_for(execution_range, sycl::detail::device_kernel_w_linear{ cgh, w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset });
    });
    detail::device_synchronize(device);

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const std::size_t num_classes = alpha_d.shape().x;
    const std::size_t num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const std::size_t num_features = predict_points_d.shape().y;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // define block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

    if (params.kernel_type == kernel_function_type::linear) {
        const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                     static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
        const ::sycl::nd_range<2> execution_range{ grid, block };

        device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
            cgh.parallel_for(execution_range, sycl::detail::device_kernel_predict_linear{ cgh, out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features });
        });
    } else {
        const std::size_t num_sv = sv_or_w_d.shape().x;

        const ::sycl::range<2> grid{ static_cast<std::size_t>(std::ceil(static_cast<double>(num_sv) / static_cast<double>(block[0] * INTERNAL_BLOCK_SIZE))) * block[0],
                                     static_cast<std::size_t>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block[1] * INTERNAL_BLOCK_SIZE))) * block[1] };
        const ::sycl::nd_range<2> execution_range{ grid, block };

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::polynomial, decltype(params.degree), real_type, decltype(params.coef0)>;
                    cgh.parallel_for(execution_range, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.degree, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::rbf:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::rbf, real_type>;
                    cgh.parallel_for(execution_range, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::sigmoid:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::sigmoid, real_type, decltype(params.coef0)>;
                    cgh.parallel_for(execution_range, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, std::get<real_type>(params.gamma), params.coef0 });
                });
                break;
            case kernel_function_type::laplacian:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::laplacian, real_type>;
                    cgh.parallel_for(execution_range, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, std::get<real_type>(params.gamma) });
                });
                break;
            case kernel_function_type::chi_squared:
                device.impl->sycl_queue.submit([&](::sycl::handler &cgh) {
                    using functor_type = sycl::detail::device_kernel_predict<kernel_function_type::chi_squared, real_type>;
                    cgh.parallel_for(execution_range, functor_type{ cgh, out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, std::get<real_type>(params.gamma) });
                });
                break;
        }
    }
    detail::device_synchronize(device);

    return out_d;
}

}  // namespace plssvm::dpcpp
