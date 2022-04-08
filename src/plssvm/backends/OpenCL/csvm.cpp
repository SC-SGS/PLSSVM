/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/OpenCL/detail/kernel.hpp"         // plssvm::opencl::detail::compute_kernel_name, plssvm::opencl::detail::kernel
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // plssvm::opencl::detail::get_contexts, plssvm::opencl::detail::create_command_queues, plssvm::opencl::detail::run_kernel, plssvm::opencl::detail::kernel_type_to_function_name, plssvm::opencl::detail::device_synchronize
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/backends/gpu_csvm.hpp"                     // plssvm::detail::gpu_csvm
#include "plssvm/constants.hpp"                             // plssvm::kernel_index_type
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"                // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "fmt/chrono.h"   // can directly print std::chrono literals
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>  // std::all_of
#include <chrono>     // std::chrono
#include <exception>  // std::terminate
#include <string>     // std::string
#include <tuple>      // std::tie
#include <utility>    // std::pair, std::make_pair, std::move
#include <vector>     // std::vector

namespace plssvm::opencl {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    base_type{ params } {
    // check whether the requested target platform has been enabled
    switch (target_) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
    }

    // get all available OpenCL contexts for the current target including devices with respect to the requested target platform
    target_platform used_target;
    std::tie(contexts_, used_target) = detail::get_contexts(target_);

    // currently, only a single context is allowed
    if (contexts_.size() != 1) {
        throw backend_exception{ fmt::format("Currently only a single OpenCL context is allowed, but {} were given!", contexts_.size()) };
    }

    // throw exception if no devices for the requested target could be found
    if (contexts_[0].devices.empty()) {
        throw backend_exception{ fmt::format("OpenCL backend selected but no devices for the target {} were found!", target_) };
    }

    // print OpenCL info
    if (print_info_) {
        fmt::print("Using OpenCL as backend.\n");
        if (target_ == target_platform::automatic) {
            fmt::print("Using {} as automatic target platform.\n", used_target);
        }
        fmt::print("\n");
    }

    // create command_queues and JIT compile OpenCL kernels
    auto jit_start_time = std::chrono::steady_clock::now();

    // get kernel names
    const std::vector<std::pair<detail::compute_kernel_name, std::string>> kernel_names = detail::kernel_type_to_function_names(kernel_);
    // the kernel order in the respective command_queue is the same as the other of the provided kernel names
    // i.e.: kernels[0] -> q_kernel, kernels[1] -> svm_kernel, kernels[2] -> w_kernel/predict_kernel
    devices_ = detail::create_command_queues<real_type>(contexts_, used_target, kernel_names, print_info_);

    auto jit_end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("OpenCL kernel JIT compilation done in {}.\n\n", std::chrono::duration_cast<std::chrono::milliseconds>(jit_end_time - jit_start_time));
    }

    // if less features than devices are provided, use only nim_features_ devices
    devices_.resize(std::min(devices_.size(), num_features_));

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // resize data vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found OpenCL devices
        fmt::print("Found {} OpenCL device(s) for the target platform {}:\n", devices_.size(), used_target);
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, detail::get_device_name(devices_[device]));
        }
        fmt::print("\n");
    }

    // sanity checks for the number of OpenCL kernels
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.size() == 3; }),
                  "Every command queue must have exactly three associated kernels!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::q_kernel) == 1; }),
                  "The q_kernel device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::svm_kernel) == 1; }),
                  "The q_kernel device kernel is missing!");
    if (kernel_ == kernel_type::linear) {
        PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::w_kernel) == 1; }),
                      "The w_kernel device kernel is missing!");
    } else {
        PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::predict_kernel) == 1; }),
                      "The predict_kernel device kernel is missing!");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the OpenCL devices have finished before destruction
        for (const queue_type &queue : devices_) {
            detail::device_synchronize(queue);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("{}\n", e.what_with_loc());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::device_synchronize(queue_type &queue) {
    detail::device_synchronize(queue);
}

std::pair<std::vector<std::size_t>, std::vector<std::size_t>> execution_range_to_native(const ::plssvm::detail::execution_range &range) {
    std::vector<std::size_t> grid = { range.grid[0], range.grid[1], range.grid[2] };
    std::vector<std::size_t> block = { range.block[0], range.block[1], range.block[2] };
    for (typename std::vector<std::size_t>::size_type i = 0; i < grid.size(); ++i) {
        grid[i] *= block[i];
    }
    return std::make_pair(std::move(grid), std::move(block));
}

template <typename T>
void csvm<T>::run_q_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &q_d, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::q_kernel], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_features));
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::q_kernel], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_cols_), degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::q_kernel], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_cols_), gamma_);
            break;
    }
}

template <typename T>
void csvm<T>::run_svm_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::svm_kernel], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_features), add, static_cast<kernel_index_type>(device));
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::svm_kernel], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_cols_), add, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::svm_kernel], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, static_cast<kernel_index_type>(num_rows_), static_cast<kernel_index_type>(num_cols_), add, gamma_);
            break;
    }
}

template <typename T>
void csvm<T>::run_w_kernel(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type &w_d, const device_ptr_type &alpha_d, const std::size_t num_features) {
    auto [grid, block] = execution_range_to_native(range);

    detail::run_kernel(devices_[device], devices_[device].kernels[detail::compute_kernel_name::w_kernel], grid, block, w_d.get(), data_d_[device].get(), data_last_d_[device].get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points_), static_cast<kernel_index_type>(num_features));
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const std::size_t num_predict_points) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            detail::run_kernel(devices_[0], devices_[0].kernels[detail::compute_kernel_name::predict_kernel], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points_), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features_), degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            detail::run_kernel(devices_[0], devices_[0].kernels[detail::compute_kernel_name::predict_kernel], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points_), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features_), gamma_);
            break;
    }
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::opencl
