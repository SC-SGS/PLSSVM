/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"     // plssvm::opencl::detail::device_ptr, plssvm::opencl::detail::device_synchronize
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // plssvm::opencl::detail::create_kernel, plssvm::opencl::detail::run_kernel, plssvm::opencl::detail::kernel_type_to_function_name
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/backends/gpu_csvm.hpp"                     // plssvm::detail::gpu_csvm
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/detail/execution_range.hpp"                // plssvm::detail::execution_range
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/target_platform.hpp"                       // plssvm::target_platform

#include "fmt/core.h"  // fmt::print, fmt::format

#include <exception>  // std::terminate
#include <string>     // std::string
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

    if (print_info_) {
        fmt::print("Using OpenCL as backend.\n");
    }

    // get all available devices wrt the requested target platform
    devices_ = detail::get_command_queues(target_);
    devices_.resize(std::min(devices_.size(), num_features_));

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("OpenCL backend selected but no devices for the target {} were found!", target_) };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found OpenLC devices
        fmt::print("Found {} OpenCL device(s) for the target platform {}:\n", devices_.size(), target_);
        for (size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, detail::get_device_name(devices_[device]));
        }
        fmt::print("\n");
    }

    // get kernel names
    std::pair<std::string, std::string> kernel_names = detail::kernel_type_to_function_name(kernel_);
    // build necessary kernel
    q_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "q_kernel.cl", kernel_names.first);
    // assemble kernel name
    svm_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "svm_kernel.cl", kernel_names.second);

    switch (kernel_) {
        case kernel_type::linear:
            kernel_w_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict_kernel.cl", "device_kernel_w_linear");
            break;
        case kernel_type::polynomial:
            predict_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict_kernel.cl", "device_kernel_predict_poly");
            break;
        case kernel_type::rbf:
            predict_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict_kernel.cl", "device_kernel_predict_radial");
            break;
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the OpenCL devices have finished before destruction
        for (const detail::command_queue &queue : devices_) {
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

template <typename size_type>
std::pair<std::vector<size_type>, std::vector<size_type>> execution_range_to_native(const ::plssvm::detail::execution_range<size_type> &range) {
    std::vector<size_type> grid = { range.grid[0], range.grid[1], range.grid[2] };
    std::vector<size_type> block = { range.block[0], range.block[1], range.block[2] };
    for (size_type i = 0; i < grid.size(); ++i) {
        grid[i] *= block[i];
    }
    return std::make_pair(std::move(grid), std::move(block));
}

template <typename T>
void csvm<T>::run_q_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &q_d, const int col_range) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, col_range);
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d.get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_);
            break;
    }
}

template <typename T>
void csvm<T>::run_svm_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const int first_feature, const int last_feature) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, last_feature - first_feature, add, static_cast<int>(device));
            break;
        case kernel_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d_[device].get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_);
            break;
    }
}

template <typename T>
void csvm<T>::run_w_kernel(const size_type device, const ::plssvm::detail::execution_range<size_type> &range, const device_ptr_type &alpha_d, const size_type num_features) {
    auto [grid, block] = execution_range_to_native(range);

    detail::run_kernel(devices_[device], kernel_w_kernel_[device], grid, block, w_d_.get(), data_d_[device].get(), data_last_d_[device].get(), alpha_d.get(), num_data_points_, num_features);
}

template <typename T>
void csvm<T>::run_predict_kernel(const ::plssvm::detail::execution_range<size_type> &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const size_type num_predict_points) {
    auto [grid, block] = execution_range_to_native(range);

    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            detail::run_kernel(devices_[0], predict_kernel_[0], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            detail::run_kernel(devices_[0], predict_kernel_[0], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), num_predict_points, num_features_, gamma_);
            break;
    }
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::opencl
