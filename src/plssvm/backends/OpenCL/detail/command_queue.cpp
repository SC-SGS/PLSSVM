/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"

#include "plssvm/backends/OpenCL/detail/error_code.hpp"
#include "plssvm/backends/OpenCL/detail/kernel.hpp"
#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

#include <memory>   // std::addressof
#include <utility>  // std::exchange

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
    queue{ std::exchange(other.queue, nullptr) }, kernels{ std::move(other.kernels) } {}

command_queue &command_queue::operator=(command_queue &&other) {
    if (this != std::addressof(other)) {
        queue = std::exchange(other.queue, nullptr);
        kernels = std::move(other.kernels);
    }
    return *this;
}

command_queue::~command_queue() {
    if (queue) {
        clReleaseCommandQueue(queue);
    }
}

}  // namespace plssvm::opencl::detail