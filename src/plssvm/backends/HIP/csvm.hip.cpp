/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                    // plssvm::backend_type
#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"                               // plssvm::hip::detail::device_ptr
#include "plssvm/backends/HIP/detail/utility.hip.hpp"                                  // PLSSVM_HIP_ERROR_CHECK, plssvm::hip::detail::{device_synchronize, get_device_count, set_device, peek_at_last_error, get_runtime_version}
#include "plssvm/backends/HIP/exceptions.hpp"                                          // plssvm::hip::backend_exception
#include "plssvm/backends/HIP/kernel/cg_explicit/blas.hip.hpp"                         // plssvm::hip::detail::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/HIP/kernel/cg_explicit/kernel_matrix_assembly.hip.hpp"       // plssvm::hip::detail::device_kernel_assembly
#include "plssvm/backends/HIP/kernel/cg_implicit/kernel_matrix_assembly_blas.hip.hpp"  // plssvm::hip::detail::device_kernel_assembly_symm
#include "plssvm/backends/HIP/kernel/predict_kernel.hip.hpp"                           // plssvm::hip::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/constants.hpp"                                                        // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                                                    // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                                                   // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                               // plssvm::detail::memory_size
#include "plssvm/detail/performance_tracker.hpp"                                       // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                                            // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                            // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                                        // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/shape.hpp"                                                            // plssvm::shape
#include "plssvm/target_platforms.hpp"                                                 // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                                 // plssvm::verbosity_level

#include "hip/hip_runtime_api.h"  // HIP runtime functions

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota
#include <utility>    // std::pair, std::make_pair

namespace plssvm::hip {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
}

void csvm::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::gpu_amd) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the HIP backend!", target) };
    } else {
#if !defined(PLSSVM_HAS_AMD_TARGET)
        throw backend_exception{ fmt::format("Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target) };
#endif
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing HIP ({}) as backend.\n",
                        plssvm::detail::tracking_entry{ "dependencies", "hip_runtime_version", detail::get_runtime_version() });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::hip }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::gpu_amd }));

    // update the target platform
    target_ = plssvm::target_platform::gpu_amd;

    // get all available devices wrt the requested target platform
    devices_.resize(detail::get_device_count());
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no HIP devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "HIP backend selected but no HIP capable devices were found!" };
    }

    // print found HIP devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} HIP device(s):\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() });
    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (const queue_type &device : devices_) {
        hipDeviceProp_t prop{};
        PLSSVM_HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, device))
        plssvm::detail::log(verbosity_level::full,
                            "  [{}, {}, {}.{}]\n",
                            device,
                            prop.name,
                            prop.major,
                            prop.minor);
        device_names.emplace_back(prop.name);
    }
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "device", device_names }));
    plssvm::detail::log(verbosity_level::full | verbosity_level::timing,
                        "\n");
}

csvm::~csvm() {
    try {
        // be sure that all operations on the HIP devices have finished before destruction
        for (const queue_type &device : devices_) {
            detail::device_synchronize(device);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    hipDeviceProp_t prop{};
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        PLSSVM_HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, devices_[device_id]))
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(prop.totalGlobalMem) };
    }
    return res;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    return this->get_device_memory();
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    hipDeviceProp_t prop{};
    PLSSVM_HIP_ERROR_CHECK(hipGetDeviceProperties(&prop, devices_[device_id]))
    return static_cast<std::size_t>(prop.maxThreadsPerBlock);
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const std::size_t device_id, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.shape().x - 1;
    const unsigned long long num_features = data_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced - row_offset) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    // calculate the number of matrix entries
    const ::plssvm::detail::triangular_data_distribution &dist = dynamic_cast<::plssvm::detail::triangular_data_distribution &>(*data_distribution_.get());
    const std::size_t num_entries_padded = dist.calculate_explicit_kernel_matrix_num_entries_padded(device_id);

    device_ptr_type kernel_matrix_d{ num_entries_padded, device };  // only explicitly store the upper triangular matrix
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    detail::set_device(device);
    switch (params.kernel_type.value()) {
        case kernel_function_type::linear:
            detail::device_kernel_assembly<kernel_function_type::linear><<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor);
            break;
        case kernel_function_type::polynomial:
            detail::device_kernel_assembly<kernel_function_type::polynomial><<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            detail::device_kernel_assembly<kernel_function_type::rbf><<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const std::size_t device_id, const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const unsigned long long num_rhs = B_d.shape().x;
    const unsigned long long num_rows = B_d.shape().y;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }

    detail::set_device(device);
    {
        const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
        const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                        static_cast<int>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

        detail::device_kernel_symm<<<grid, block>>>(num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get());
    }

    {
        const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

        if (num_mirror_rows > 0) {
            const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
            const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                            static_cast<int>(std::ceil(static_cast<double>(num_mirror_rows) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

            detail::device_kernel_symm_mirror<<<grid, block>>>(num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get());
        }
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const unsigned long long num_rows = lhs_d.shape().y;
    const queue_type &device = devices_[device_id];

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    detail::set_device(device);
    detail::device_kernel_inplace_matrix_add<<<grid, block>>>(num_rhs, lhs_d.get(), rhs_d.get());
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, device_ptr_type &lhs_d, const real_type scale) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const unsigned long long num_rows = lhs_d.shape().y;
    const queue_type &device = devices_[device_id];

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    detail::set_device(device);
    detail::device_kernel_inplace_matrix_scale<<<grid, block>>>(num_rhs, lhs_d.get(), scale);
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const std::size_t device_id, const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const unsigned long long num_rows_reduced = A_d.shape().x - 1;
    const unsigned long long num_features = A_d.shape().y;
    const unsigned long long num_classes = B_d.shape().x;
    const queue_type &device = devices_[device_id];

    // calculate the number of data points this device is responsible for
    const unsigned long long device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
    // get the offset of the data points this device is responsible for
    const unsigned long long row_offset = data_distribution_->place_row_offset(device_id);

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced - row_offset) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    detail::set_device(device);
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type.value()) {
        case kernel_function_type::linear:
            detail::device_kernel_assembly_symm<kernel_function_type::linear><<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes);
            break;
        case kernel_function_type::polynomial:
            detail::device_kernel_assembly_symm<kernel_function_type::polynomial><<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            detail::device_kernel_assembly_symm<kernel_function_type::rbf><<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const std::size_t device_id, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = alpha_d.shape().y;
    const unsigned long long device_specific_num_sv = sv_d.shape().x;
    const unsigned long long num_features = sv_d.shape().y;
    const queue_type &device = devices_[device_id];

    // get the offset of the data points this device is responsible for
    const unsigned long long sv_offset = data_distribution_->place_row_offset(device_id);

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    detail::set_device(device);
    detail::device_kernel_w_linear<<<grid, block>>>(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset);
    detail::peek_at_last_error();
    detail::device_synchronize(device);

    return w_d;
}

auto csvm::run_predict_kernel(const std::size_t device_id, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_predict_points = predict_points_d.shape().x;  // = device_specific_num_rows
    const unsigned long long num_features = predict_points_d.shape().y;
    const queue_type &device = devices_[device_id];

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, device };

    // define the block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size(device_id);
    if (max_work_group_size < std::size_t{ THREAD_BLOCK_SIZE } * std::size_t{ THREAD_BLOCK_SIZE }) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);

    detail::set_device(device);
    if (params.kernel_type == kernel_function_type::linear) {
        // define the grid sizes
        const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                        static_cast<int>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

        detail::device_kernel_predict_linear<<<grid, block>>>(out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features);
    } else {
        const unsigned long long num_sv = sv_or_w_d.shape().x;

        // define the grid sizes
        const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                        static_cast<int>(std::ceil(static_cast<double>(num_sv) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

        switch (params.kernel_type.value()) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                detail::device_kernel_predict<kernel_function_type::polynomial><<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                detail::device_kernel_predict<kernel_function_type::rbf><<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.gamma.value());
                break;
        }
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);

    return out_d;
}

}  // namespace plssvm::hip
