/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backend_types.hpp"                                          // plssvm::backend_type
#include "plssvm/backends/CUDA/cg_explicit/blas.cuh"                         // plssvm::cuda::device_kernel_gemm
#include "plssvm/backends/CUDA/cg_explicit/kernel_matrix_assembly.cuh"       // plssvm::cuda::{device_kernel_assembly_linear, device_kernel_assembly_polynomial, device_kernel_assembly_rbf}
#include "plssvm/backends/CUDA/cg_implicit/kernel_matrix_assembly_blas.cuh"  // plssvm::cuda::device_kernel_assembly_linear_symm, plssvm::cuda::device_kernel_assembly_polynomial_symm, plssvm::cuda::device_kernel_assembly_rbf_symm
#include "plssvm/backends/CUDA/detail/device_ptr.cuh"                        // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"                           // plssvm::cuda::detail::{device_synchronize, get_device_count, set_device, peek_at_last_error, get_runtime_version}
#include "plssvm/backends/CUDA/exceptions.hpp"                               // plssvm::cuda::backend_exception
#include "plssvm/backends/CUDA/predict_kernel.cuh"                           // plssvm::cuda::detail::{device_kernel_w_linear, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/constants.hpp"                                              // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/assert.hpp"                                          // PLSSVM_ASSERT
#include "plssvm/detail/logging.hpp"                                         // plssvm::detail::log
#include "plssvm/detail/memory_size.hpp"                                     // plssvm::detail::memory_size
#include "plssvm/detail/performance_tracker.hpp"                             // plssvm::detail::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                                  // plssvm::exception
#include "plssvm/kernel_function_types.hpp"                                  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                                              // plssvm::parameter
#include "plssvm/shape.hpp"                                                  // plssvm::shape
#include "plssvm/target_platforms.hpp"                                       // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                                       // plssvm::verbosity_level

#include "cuda.h"              // cuda runtime functions
#include "cuda_runtime_api.h"  // cuda runtime functions

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cmath>      // std::sqrt, std::ceil
#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota
#include <string>     // std::string
#include <vector>     // std:vector

namespace plssvm::cuda {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

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
    devices_.resize(detail::get_device_count());
    std::iota(devices_.begin(), devices_.end(), 0);

    // currently only single GPU execution is supported
    if (devices_.size() > 1) {
        plssvm::detail::log(verbosity_level::full | verbosity_level::warning,
                            "WARNING: found {} devices, but currently only single GPU execution is supported. Continuing only with device 0!\n",
                            devices_.size());
        devices_.resize(1);
    }

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
        cudaGetDeviceProperties(&prop, device);
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

::plssvm::detail::memory_size csvm::get_device_memory() const {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, devices_[0]);
    return ::plssvm::detail::memory_size{ static_cast<unsigned long long>(prop.totalGlobalMem) };
}

::plssvm::detail::memory_size csvm::get_max_mem_alloc_size() const {
    return this->get_device_memory();
}

std::size_t csvm::get_max_work_group_size() const {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, devices_[0]);
    return static_cast<std::size_t>(prop.maxThreadsPerBlock);
}

//***************************************************//
//                        fit                        //
//***************************************************//

auto csvm::run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const -> device_ptr_type {
    const unsigned long long num_rows_reduced = data_d.shape().x - 1;
    const unsigned long long num_features = data_d.shape().y;

    // define grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

#if defined(PLSSVM_USE_GEMM)
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE), devices_[0] };  // store full matrix
#else
    device_ptr_type kernel_matrix_d{ (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2, devices_[0] };  // only explicitly store the upper triangular matrix
#endif
    kernel_matrix_d.memset(0);
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    detail::set_device(0);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            cuda::device_kernel_assembly_linear<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor);
            break;
        case kernel_function_type::polynomial:
            cuda::device_kernel_assembly_polynomial<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            cuda::device_kernel_assembly_rbf<<<grid, block>>>(kernel_matrix_d.get(), data_d.get(), num_rows_reduced, num_features, q_red_d.get(), QA_cost, cost_factor, params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
    detail::device_synchronize(devices_[0]);

    return kernel_matrix_d;
}

void csvm::run_blas_level_3_kernel_explicit(const real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, const real_type beta, device_ptr_type &C_d) const {
    const unsigned long long num_rhs = B_d.shape().x;
    const unsigned long long num_rows = B_d.shape().y;

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rhs) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_rows) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    detail::set_device(0);
#if defined(PLSSVM_USE_GEMM)
    cuda::device_kernel_gemm<<<grid, block>>>(num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get());
#else
    cuda::device_kernel_symm<<<grid, block>>>(num_rows, num_rhs, num_rows, alpha, A_d.get(), B_d.get(), beta, C_d.get());
#endif
    detail::peek_at_last_error();
    detail::device_synchronize(devices_[0]);
}

void csvm::run_assemble_kernel_matrix_implicit_blas_level_3(const real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red, const real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const {
    const unsigned long long num_rows_reduced = A_d.shape().x - 1;
    const unsigned long long num_features = A_d.shape().y;
    const unsigned long long num_classes = B_d.shape().x;

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_rows_reduced) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    detail::set_device(0);
    const real_type cost_factor = real_type{ 1.0 } / params.cost;

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            cuda::device_kernel_assembly_linear_symm<<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, B_d.get(), C_d.get(), num_classes);
            break;
        case kernel_function_type::polynomial:
            cuda::device_kernel_assembly_polynomial_symm<<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, params.degree.value(), params.gamma.value(), params.coef0.value(), B_d.get(), C_d.get(), num_classes);
            break;
        case kernel_function_type::rbf:
            cuda::device_kernel_assembly_rbf_symm<<<grid, block>>>(alpha, q_red.get(), A_d.get(), num_rows_reduced, num_features, QA_cost, cost_factor, params.gamma.value(), B_d.get(), C_d.get(), num_classes);
            break;
    }
    detail::peek_at_last_error();
    detail::device_synchronize(devices_[0]);
}

//***************************************************//
//                   predict, score                  //
//***************************************************//

auto csvm::run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = sv_d.shape().x;
    const unsigned long long num_features = sv_d.shape().y;

    // define the grid and block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
    const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_features) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                    static_cast<int>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

    device_ptr_type w_d{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    detail::set_device(0);
    cuda::device_kernel_w_linear<<<grid, block>>>(w_d.get(), alpha_d.get(), sv_d.get(), num_classes, num_sv);
    detail::peek_at_last_error();
    detail::device_synchronize(devices_[0]);

    return w_d;
}

auto csvm::run_predict_kernel(const parameter &params, const device_ptr_type &w_d, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const -> device_ptr_type {
    const unsigned long long num_classes = alpha_d.shape().x;
    const unsigned long long num_sv = sv_d.shape().x;
    const unsigned long long num_predict_points = predict_points_d.shape().x;
    const unsigned long long num_features = predict_points_d.shape().y;

    device_ptr_type out_d{ shape{ num_predict_points, num_classes }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[0] };

    // define the block sizes
    const std::size_t max_work_group_size = this->get_max_work_group_size();
    if (max_work_group_size < THREAD_BLOCK_SIZE * THREAD_BLOCK_SIZE) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}! Try reducing THREAD_BLOCK_SIZE.", THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE) };
    }
    const dim3 block(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);

    detail::set_device(0);
    if (params.kernel_type == kernel_function_type::linear) {
        // define the grid sizes
        const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                        static_cast<int>(std::ceil(static_cast<double>(num_classes) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

        cuda::device_kernel_predict_linear<<<grid, block>>>(out_d.get(), w_d.get(), rho_d.get(), predict_points_d.get(), num_classes, num_predict_points, num_features);
    } else {
        // define the grid sizes
        const dim3 grid(static_cast<int>(std::ceil(static_cast<double>(num_predict_points) / static_cast<double>(block.x * INTERNAL_BLOCK_SIZE))),
                        static_cast<int>(std::ceil(static_cast<double>(num_sv) / static_cast<double>(block.y * INTERNAL_BLOCK_SIZE))));

        switch (params.kernel_type) {
            case kernel_function_type::linear:
                // already handled
                break;
            case kernel_function_type::polynomial:
                cuda::device_kernel_predict_polynomial<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.degree.value(), params.gamma.value(), params.coef0.value());
                break;
            case kernel_function_type::rbf:
                cuda::device_kernel_predict_rbf<<<grid, block>>>(out_d.get(), alpha_d.get(), rho_d.get(), sv_d.get(), predict_points_d.get(), num_classes, num_sv, num_predict_points, num_features, params.gamma.value());
                break;
        }
    }
    detail::peek_at_last_error();
    detail::device_synchronize(devices_[0]);

    return out_d;
}

}  // namespace plssvm::cuda
