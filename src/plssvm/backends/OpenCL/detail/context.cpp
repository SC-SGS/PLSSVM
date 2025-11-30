/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/context.hpp"

#include "CL/cl.h"  // cl_context, cl_platform_id, cl_device_id, clReleaseContext

#include <memory>   // std::addressof
#include <utility>  // std::exchange, std::move

namespace plssvm::opencl::detail {

context::context(cl_context p_device_context, cl_platform_id p_platform, cl_device_id p_device) :
    device_context{ p_device_context },
    platform{ p_platform },
    device{ p_device } { }

context::context(context &&other) noexcept :
    device_context{ std::exchange(other.device_context, cl_context{}) },
    platform{ std::exchange(other.platform, cl_platform_id{}) },
    device{ std::exchange(other.device, cl_device_id{}) } { }

context &context::operator=(context &&other) noexcept {
    if (this != std::addressof(other)) {
        other.device_context = std::exchange(other.device_context, cl_context{});
        platform = std::exchange(other.platform, cl_platform_id{});
        device = std::exchange(other.device, cl_device_id{});
    }
    return *this;
}

context::~context() {
    if (device_context) {
        clReleaseContext(device_context);
    }
}

}  // namespace plssvm::opencl::detail
