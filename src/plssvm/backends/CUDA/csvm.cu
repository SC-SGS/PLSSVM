/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/backends/CUDA/detail/utility.cuh"     // plssvm::cuda::detail::{device_synchronize, get_device_count, set_device, peek_at_last_error}
#include "plssvm/backends/CUDA/exceptions.hpp"         // plssvm::cuda::backend_exception
#include "plssvm/backends/CUDA/predict_kernel.cuh"     // plssvm::cuda::detail::{device_kernel_w_linear, device_kernel_predict_polynomial, device_kernel_predict_rbf}
#include "plssvm/backends/CUDA/q_kernel.cuh"           // plssvm::cuda::detail::{device_kernel_q_linear, device_kernel_q_polynomial, device_kernel_q_rbf}
#include "plssvm/backends/CUDA/svm_kernel.cuh"         // plssvm::cuda::detail::{device_kernel_linear, device_kernel_polynomial, device_kernel_rbf}
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"           // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/kernel_function_types.hpp"            // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter, plssvm::detail::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "cuda.h"              // cuda runtime functions
#include "cuda_runtime_api.h"  // cuda runtime functions

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <numeric>    // std::iota
#include <utility>    // std::pair, std::make_pair

namespace plssvm::cuda {

csvm::csvm(parameter params) :
    csvm{ plssvm::target_platform::automatic, params } {}

csvm::csvm(target_platform target, parameter params) :
    base_type{ params } {
    this->init(target);
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

    if (plssvm::verbose) {
        std::cout << fmt::format("\nUsing CUDA as backend.") << std::endl;
    }

    // get all available devices wrt the requested target platform
    devices_.resize(detail::get_device_count());
    std::iota(devices_.begin(), devices_.end(), 0);

    // throw exception if no CUDA devices could be found
    if (devices_.empty()) {
        throw backend_exception{ "CUDA backend selected but no CUDA capable devices were found!" };
    }

    if (plssvm::verbose) {
        // print found CUDA devices
        std::cout << fmt::format("Found {} CUDA device(s):\n", devices_.size());
        for (const queue_type &device : devices_) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            std::cout << fmt::format("  [{}, {}, {}.{}]\n", device, prop.name, prop.major, prop.minor) << std::endl;
        }
        std::cout << std::endl;
    }
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

void csvm::device_synchronize(const queue_type &queue) const {
    detail::device_synchronize(queue);
}

std::pair<dim3, dim3> execution_range_to_native(const ::plssvm::detail::execution_range &range) {
    dim3 grid(range.grid[0], range.grid[1], range.grid[2]);
    dim3 block(range.block[0], range.block[1], range.block[2]);
    return std::make_pair(grid, block);
}

template <typename real_type>
void csvm::run_q_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &q_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    const auto [grid, block] = execution_range_to_native(range);

    detail::set_device(static_cast<queue_type>(device));
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            cuda::device_kernel_q_linear<<<grid, block>>>(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features));
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            cuda::device_kernel_q_polynomial<<<grid, block>>>(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            cuda::device_kernel_q_rbf<<<grid, block>>>(q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
}

template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_svm_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const real_type QA_cost, const real_type add, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    const auto [grid, block] = execution_range_to_native(range);

    detail::set_device(static_cast<queue_type>(device));
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            cuda::device_kernel_linear<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, static_cast<kernel_index_type>(device));
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            cuda::device_kernel_polynomial<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            cuda::device_kernel_rbf<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
}

template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, const device_ptr_type<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, float, float, std::size_t, std::size_t) const;
template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, const device_ptr_type<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, double, double, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_w_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points, const std::size_t num_features) const {
    const auto [grid, block] = execution_range_to_native(range);

    detail::set_device(static_cast<queue_type>(device));
    cuda::device_kernel_w_linear<<<grid, block>>>(w_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points), static_cast<kernel_index_type>(num_features));
    detail::peek_at_last_error();
}

template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &out_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &point_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_support_vectors, const std::size_t num_predict_points, const std::size_t num_features) const {
    const auto [grid, block] = execution_range_to_native(range);

    detail::set_device(0);
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            cuda::device_kernel_predict_polynomial<<<grid, block>>>(out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.degree.value(), params.gamma.value(), params.coef0.value());
            break;
        case kernel_function_type::rbf:
            cuda::device_kernel_predict_rbf<<<grid, block>>>(out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.gamma.value());
            break;
    }
    detail::peek_at_last_error();
}

template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t, std::size_t) const;
template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t, std::size_t) const;

}  // namespace plssvm::cuda
