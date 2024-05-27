/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backend_types.hpp"                                                 // plssvm::backend_type
#include "plssvm/backends/CUDA/detail/device_ptr.cuh"                               // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"                                  // PLSSVM_CUDA_ERROR_CHECK, plssvm::cuda::detail::{dim_type_to_native, device_synchronize, get_device_count, set_device, peek_at_last_error, get_runtime_version}
#include "plssvm/backends/CUDA/exceptions.hpp"                                      // plssvm::cuda::backend_exception
#include "plssvm/backends/CUDA/kernel/cg_explicit/blas.cuh"                         // plssvm::cuda::detail::{device_kernel_symm, device_kernel_symm_mirror, device_kernel_inplace_matrix_add, device_kernel_inplace_matrix_scale}
#include "plssvm/backends/CUDA/kernel/cg_explicit/kernel_matrix_assembly.cuh"       // plssvm::cuda::detail::device_kernel_assembly
#include "plssvm/backends/CUDA/kernel/cg_implicit/kernel_matrix_assembly_blas.cuh"  // plssvm::cuda::detail::device_kernel_assembly_symm
#include "plssvm/backends/CUDA/kernel/predict_kernel.cuh"                           // plssvm::cuda::detail::{device_kernel_w_linear, device_kernel_predict_linear, device_kernel_predict}
#include "plssvm/backends/execution_range.hpp"                                      // plssvm::detail::{dim_type, execution_range}
#include "plssvm/constants.hpp"                                                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                                                 // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"                                      // plssvm::detail::{data_distribution, triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/logging.hpp"                                                // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                            // plssvm::detail::memory_size
#include "plssvm/detail/performance_tracker.hpp"                                    // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                                         // plssvm::exception
#include "plssvm/gamma.hpp"                                                         // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"                                         // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                                     // plssvm::parameter
#include "plssvm/shape.hpp"                                                         // plssvm::shape
#include "plssvm/target_platforms.hpp"                                              // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                              // plssvm::verbosity_level

#include "cuda.h"              // cuda runtime
#include "cuda_runtime.h"      // cuda runtime
#include "cuda_runtime_api.h"  // cuda runtime functions

#include "fmt/core.h"  // fmt::format

#include <cmath>      // std::sqrt, std::ceil
#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota
#include <string>     // std::string
#include <variant>    // std::get
#include <vector>     // std:vector

namespace plssvm::cuda {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } { }

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
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

void csvm::init(const target_platform target) {
    // check if supported target platform has been selected
    if (target != target_platform::automatic && target != target_platform::gpu_nvidia) {
        throw backend_exception{ fmt::format("Invalid target platform '{}' for the CUDA backend!", target) };
    } else {
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
        throw backend_exception{ "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!" };
#endif
    }

    plssvm::detail::log(verbosity_level::full,
                        "\nUsing CUDA ({}) as backend.\n",
                        plssvm::detail::tracking_entry{ "dependencies", "cuda_runtime_version", detail::get_runtime_version() });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "backend", plssvm::backend_type::cuda }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "backend", "target_platform", plssvm::target_platform::gpu_nvidia }));

    // update the target platform
    target_ = plssvm::target_platform::gpu_nvidia;

    // get all available devices wrt the requested target platform
    devices_.resize(static_cast<std::vector<queue_type>::size_type>(detail::get_device_count()));
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "CUDA backend selected but no CUDA capable devices were found!" };
    }

    // print found CUDA devices
    plssvm::detail::log(verbosity_level::full,
                        "Found {} CUDA device(s):\n",
                        plssvm::detail::tracking_entry{ "backend", "num_devices", devices_.size() });
    std::vector<std::string> device_names;
    device_names.reserve(devices_.size());
    for (const queue_type &device : devices_) {
        cudaDeviceProp prop{};
        PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, device))
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

std::vector<::plssvm::detail::memory_size> csvm::get_device_memory() const {
    cudaDeviceProp prop{};
    std::vector<::plssvm::detail::memory_size> res(this->num_available_devices());
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, devices_[device_id]))
        res[device_id] = ::plssvm::detail::memory_size{ static_cast<unsigned long long>(prop.totalGlobalMem) };
    }
    return res;
}

std::vector<::plssvm::detail::memory_size> csvm::get_max_mem_alloc_size() const {
    return this->get_device_memory();
}

std::size_t csvm::get_max_work_group_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    cudaDeviceProp prop{};
    PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, devices_[device_id]))
    return static_cast<std::size_t>(prop.maxThreadsPerBlock);
}

::plssvm::detail::dim_type csvm::get_max_grid_size(const std::size_t device_id) const {
    PLSSVM_ASSERT(device_id < this->num_available_devices(), "Invalid device {} requested!", device_id);
    cudaDeviceProp prop{};
    PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, devices_[device_id]));
     return { static_cast<std::size_t>(prop.maxGridSize[0]), static_cast<std::size_t>(prop.maxGridSize[1]), static_cast<std::size_t>(prop.maxGridSize[2]) };
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

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::device_kernel_assembly<kernel_function_type::linear><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y);
                break;
            case kernel_function_type::polynomial:
                detail::device_kernel_assembly<kernel_function_type::polynomial><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::device_kernel_assembly<kernel_function_type::rbf><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::device_kernel_assembly<kernel_function_type::sigmoid><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::device_kernel_assembly<kernel_function_type::laplacian><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::device_kernel_assembly<kernel_function_type::chi_squared><<<native_partial_grid, native_block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, q_red_d.get(), QA_cost, cost_factor, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::peek_at_last_error();
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

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (std::size_t i = 0; i < exec.grids.size(); ++i) {
        {
            const auto [partial_grid, offsets] = exec.grids[i];
            // convert execution range grid[i] to CUDA's native dim3
            const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

            detail::device_kernel_symm<<<native_partial_grid, native_block>>>(num_rows, num_rhs, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y);
        }

        {
            const unsigned long long num_mirror_rows = num_rows - row_offset - device_specific_num_rows;

            if (num_mirror_rows > 0) {
                const auto [partial_grid, offsets] = mirror_exec.grids[i];
                // convert execution range grid[i] to CUDA's native dim3
                const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

                detail::device_kernel_symm_mirror<<<native_partial_grid, native_block>>>(num_rows, num_rhs, num_mirror_rows, device_specific_num_rows, row_offset, alpha, A_d.get(), B_d.get(), beta, C_d.get(), offsets.x, offsets.y);
            }
        }
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_addition(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        detail::device_kernel_inplace_matrix_add<<<native_partial_grid, native_block>>>(num_rhs, lhs_d.get(), rhs_d.get(), offsets.x, offsets.y);
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);
}

void csvm::run_inplace_matrix_scale(const std::size_t device_id, const ::plssvm::detail::execution_range &exec, device_ptr_type &lhs_d, const real_type scale) const {
    const unsigned long long num_rhs = lhs_d.shape().x;
    const queue_type &device = devices_[device_id];

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        detail::device_kernel_inplace_matrix_scale<<<native_partial_grid, native_block>>>(num_rhs, lhs_d.get(), scale, offsets.x, offsets.y);
    }
    detail::peek_at_last_error();
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

    // convert general execution range's block to CUDA specific block
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::device_kernel_assembly_symm<kernel_function_type::linear><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y);
                break;
            case kernel_function_type::polynomial:
                detail::device_kernel_assembly_symm<kernel_function_type::polynomial><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::device_kernel_assembly_symm<kernel_function_type::rbf><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::device_kernel_assembly_symm<kernel_function_type::sigmoid><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::device_kernel_assembly_symm<kernel_function_type::laplacian><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::device_kernel_assembly_symm<kernel_function_type::chi_squared><<<native_partial_grid, native_block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, device_specific_num_rows, row_offset, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::peek_at_last_error();
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

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        detail::device_kernel_w_linear<<<native_partial_grid, native_block>>>(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv, device_specific_num_sv, sv_offset, offsets.x, offsets.y);
    }
    detail::peek_at_last_error();
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

    // convert execution range block to CUDA's native dim3
    const dim3 native_block = detail::dim_type_to_native(exec.block);

    detail::set_device(device);
    for (const auto &grid : exec.grids) {
        const auto [partial_grid, offsets] = grid;
        // convert execution range grid[i] to CUDA's native dim3
        const dim3 native_partial_grid = detail::dim_type_to_native(partial_grid);

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                detail::device_kernel_predict_linear<<<native_partial_grid, native_block>>>(out_d.get(), sv_or_w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features, offsets.x, offsets.y);
                break;
            case kernel_function_type::polynomial:
                detail::device_kernel_predict<kernel_function_type::polynomial><<<native_partial_grid, native_block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, params.degree, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::rbf:
                detail::device_kernel_predict<kernel_function_type::rbf><<<native_partial_grid, native_block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::sigmoid:
                detail::device_kernel_predict<kernel_function_type::sigmoid><<<native_partial_grid, native_block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma), params.coef0);
                break;
            case kernel_function_type::laplacian:
                detail::device_kernel_predict<kernel_function_type::laplacian><<<native_partial_grid, native_block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
            case kernel_function_type::chi_squared:
                detail::device_kernel_predict<kernel_function_type::chi_squared><<<native_partial_grid, native_block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_or_w_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, offsets.x, offsets.y, std::get<real_type>(params.gamma));
                break;
        }
    }
    detail::peek_at_last_error();
    detail::device_synchronize(device);

    return out_d;
}

}  // namespace plssvm::cuda
