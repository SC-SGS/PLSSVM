/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/backends/OpenCL/detail/context.hpp"

#include "plssvm/backends/OpenCL/detail/error_code.hpp"
#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace plssvm::opencl::detail {

context::context(cl_context p_device_context, cl_platform_id p_platform, std::vector<cl_device_id> p_devices) :
    device_context{ p_device_context }, platform{ p_platform }, devices{ std::move(p_devices) } {
    queues.reserve(devices.size());
    error_code err;
    for (const cl_device_id &device : devices) {
#ifdef CL_VERSION_2_0
        // use new clCreateCommandQueueWithProperties function
        queues.emplace_back(clCreateCommandQueueWithProperties(device_context, device, 0, &err));
#else
        // use old clCreateCommandQueue function (deprecated in newer OpenCL versions)
        queues.emplace_back(clCreateCommandQueue(device_context, device, 0, &err));
#endif
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL command queue");
    }
}

context::context(context &&other) noexcept :
    device_context{ std::exchange(other.device_context, nullptr) }, platform{ other.platform }, devices{ std::move(other.devices) }, queues{ std::move(other.queues) } {}

context &context::operator=(context &&other) {
    if (this != std::addressof(other)) {
        other.device_context = std::exchange(other.device_context, nullptr);
        platform = other.platform;
        devices = std::move(other.devices);
        queues = std::move(other.queues);
    }
    return *this;
}

context::~context() {
    // TODO: destruction command queue
    if (device_context) {
        clReleaseContext(device_context);
    }
}

}  // namespace plssvm::opencl::detail