/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"

#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/kernel.hpp"      // plssvm::opencl::detail::kernel
#include "plssvm/backends/OpenCL/detail/utility.hpp"     // PLSSVM_OPENCL_ERROR_CHECK
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/type_traits.hpp"                 // plssvm::detail::always_false_v

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clCreateCommandQueueWithProperties, clCreateCommandQueue, clReleaseCommandQueue

#include <memory>       // std::addressof
#include <type_traits>  // std::is_same_v
#include <utility>      // std::exchange, std::move

namespace plssvm::opencl::detail {

command_queue::command_queue(cl_context context, cl_device_id device) {
    error_code err;
#ifdef CL_VERSION_2_0
    // use new clCreateCommandQueueWithProperties function
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
#else
    // use old clCreateCommandQueue function (deprecated in newer OpenCL versions)
    queue = clCreateCommandQueue(context, device, 0, &err);
#endif
    PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL command queue");
}

command_queue::command_queue(command_queue &&other) noexcept :
    queue{ std::exchange(other.queue, nullptr) }, float_kernels{ std::move(other.float_kernels) }, double_kernels{ std::move(other.double_kernels) } {}

command_queue &command_queue::operator=(command_queue &&other) noexcept {
    if (this != std::addressof(other)) {
        queue = std::exchange(other.queue, nullptr);
        float_kernels = std::move(other.float_kernels);
        double_kernels = std::move(other.double_kernels);
    }
    return *this;
}

command_queue::~command_queue() {
    if (queue) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseCommandQueue(queue), "error releasing cl_command_queue");
    }
}

template <typename real_type>
void command_queue::add_kernel(compute_kernel_name name, kernel &&compute_kernel) {
    if constexpr (std::is_same_v<real_type, float>) {
        PLSSVM_ASSERT(float_kernels.count(name) == 0, "The given (float) kernel as already been added to this command queue!");
        float_kernels.insert_or_assign(name, std::move(compute_kernel));
    } else if constexpr (std::is_same_v<real_type, double>) {
        PLSSVM_ASSERT(double_kernels.count(name) == 0, "The given (double) kernel as already been added to this command queue!");
        double_kernels.insert_or_assign(name, std::move(compute_kernel));
    } else {
        ::plssvm::detail::always_false_v<real_type>;
    }
}

template void command_queue::add_kernel<float>(compute_kernel_name, kernel &&);
template void command_queue::add_kernel<double>(compute_kernel_name, kernel &&);

template <typename real_type>
[[nodiscard]] const kernel &command_queue::get_kernel(compute_kernel_name name) const {
    if constexpr (std::is_same_v<real_type, float>) {
        return float_kernels.at(name);
    } else if constexpr (std::is_same_v<real_type, double>) {
        return double_kernels.at(name);
    } else {
        ::plssvm::detail::always_false_v<real_type>;
    }
}

template const kernel &command_queue::get_kernel<float>(compute_kernel_name) const;
template const kernel &command_queue::get_kernel<double>(compute_kernel_name) const;

}  // namespace plssvm::opencl::detail