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
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
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

    // get kernel type from base class
    const kernel_function_type kernel = base_type::get_params().kernel_type;

    // get all available OpenCL contexts for the current target including devices with respect to the requested target platform
    target_platform used_target;
    std::tie(contexts_, used_target) = detail::get_contexts(target);

    // currently, only a single context is allowed
    if (contexts_.size() != 1) {
        throw backend_exception{ fmt::format("Currently only a single OpenCL context is allowed, but {} were given!", contexts_.size()) };
    }

    // throw exception if no devices for the requested target could be found
    if (contexts_[0].devices.empty()) {
        throw backend_exception{ fmt::format("OpenCL backend selected but no devices for the target {} were found!", target) };
    }

    // print OpenCL info
    if (verbose) {
        std::cout << fmt::format("Using OpenCL as backend.\n");
        if (target == target_platform::automatic) {
            std::cout << fmt::format("Using {} as automatic target platform.\n", used_target);
        }
        std::cout << std::endl;
    }

    // create command_queues and JIT compile OpenCL kernels
    auto jit_start_time = std::chrono::steady_clock::now();

    // get kernel names
    const std::vector<std::pair<detail::compute_kernel_name, std::string>> kernel_names = detail::kernel_type_to_function_names(kernel);
    // the kernel order in the respective command_queue is the same as the other of the provided kernel names
    // i.e.: kernels[0] -> q_kernel, kernels[1] -> svm_kernel, kernels[2] -> w_kernel/predict_kernel
    using real_type = double;  // TODO: !!!!
    devices_ = detail::create_command_queues<real_type>(contexts_, used_target, kernel_names, verbose);

    auto jit_end_time = std::chrono::steady_clock::now();
    if (verbose) {
        std::cout << fmt::format("OpenCL kernel JIT compilation done in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(jit_end_time - jit_start_time)) << std::endl;
    }

    if (verbose) {
        // print found OpenCL devices
        std::cout << fmt::format("Found {} OpenCL device(s) for the target platform {}:\n", devices_.size(), used_target);
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            std::cout << fmt::format("  [{}, {}]\n", device, detail::get_device_name(devices_[device]));
        }
        std::cout << std::endl;
    }

    // sanity checks for the number of OpenCL kernels
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.size() == 3; }),
                  "Every command queue must have exactly three associated kernels!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::q_kernel) == 1; }),
                  "The q_kernel device kernel is missing!");
    PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::svm_kernel) == 1; }),
                  "The q_kernel device kernel is missing!");
    if (kernel == kernel_function_type::linear) {
        PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::w_kernel) == 1; }),
                      "The w_kernel device kernel is missing!");
    } else {
        PLSSVM_ASSERT(std::all_of(devices_.begin(), devices_.end(), [](const queue_type &queue) { return queue.kernels.count(detail::compute_kernel_name::predict_kernel) == 1; }),
                      "The predict_kernel device kernel is missing!");
    }
}

csvm::~csvm() {
    try {
        // be sure that all operations on the OpenCL devices have finished before destruction
        for (const queue_type &queue : devices_) {
            detail::device_synchronize(queue);
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

void csvm::device_synchronize(const queue_type &queue) const {
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

template <typename real_type>
void csvm::run_q_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &q_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    auto [grid, block] = execution_range_to_native(range);

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::q_kernel), grid, block, q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features));
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::q_kernel), grid, block, q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.degree, params.gamma, params.coef0);
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The polynomial kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::q_kernel), grid, block, q_d.get(), data_d.get(), data_last_d.get(), static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), params.gamma);
            break;
    }
}

template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_q_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_svm_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const real_type QA_cost, const real_type add, const std::size_t num_data_points_padded, const std::size_t num_features) const {
    auto [grid, block] = execution_range_to_native(range);

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::svm_kernel), grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, static_cast<kernel_index_type>(device));
            break;
        case kernel_function_type::polynomial:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::svm_kernel), grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.degree, params.gamma, params.coef0);
            break;
        case kernel_function_type::rbf:
            PLSSVM_ASSERT(device == 0, "The radial basis function kernel function currently only supports single GPU execution!");
            detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::svm_kernel), grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, 1 / params.cost, static_cast<kernel_index_type>(num_data_points_padded), static_cast<kernel_index_type>(num_features), add, params.gamma);
            break;
    }
}

template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, const device_ptr_type<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, float, float, std::size_t, std::size_t) const;
template void csvm::run_svm_kernel_impl(std::size_t, const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, const device_ptr_type<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, double, double, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_w_kernel_impl(const std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<real_type> &w_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_data_points, const std::size_t num_features) const {
    auto [grid, block] = execution_range_to_native(range);

    detail::run_kernel(devices_[device], devices_[device].kernels.at(detail::compute_kernel_name::w_kernel), grid, block, w_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_data_points), static_cast<kernel_index_type>(num_features));
}

template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t) const;
template void csvm::run_w_kernel_impl(std::size_t device, const ::plssvm::detail::execution_range &range, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t) const;

template <typename real_type>
void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &range, const ::plssvm::detail::parameter<real_type> &params, device_ptr_type<real_type> &out_d, const device_ptr_type<real_type> &alpha_d, const device_ptr_type<real_type> &point_d, const device_ptr_type<real_type> &data_d, const device_ptr_type<real_type> &data_last_d, const std::size_t num_support_vectors, const std::size_t num_predict_points, const std::size_t num_features) const {
    auto [grid, block] = execution_range_to_native(range);

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            detail::run_kernel(devices_[0], devices_[0].kernels.at(detail::compute_kernel_name::predict_kernel), grid, block, out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.degree, params.gamma, params.coef0);
            break;
        case kernel_function_type::rbf:
            detail::run_kernel(devices_[0], devices_[0].kernels.at(detail::compute_kernel_name::predict_kernel), grid, block, out_d.get(), data_d.get(), data_last_d.get(), alpha_d.get(), static_cast<kernel_index_type>(num_support_vectors), point_d.get(), static_cast<kernel_index_type>(num_predict_points), static_cast<kernel_index_type>(num_features), params.gamma);
            break;
    }
}

template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<float> &, device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, const device_ptr_type<float> &, std::size_t, std::size_t, std::size_t) const;
template void csvm::run_predict_kernel_impl(const ::plssvm::detail::execution_range &, const ::plssvm::detail::parameter<double> &, device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, const device_ptr_type<double> &, std::size_t, std::size_t, std::size_t) const;

}  // namespace plssvm::opencl
