/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace plssvm::opencl::detail {

command_queue::command_queue(cl_context p_context, cl_command_queue p_queue, cl_device_id p_device) :
    context{ p_context }, queue{ p_queue }, device{ p_device } {}

command_queue::command_queue(command_queue &&other) noexcept :
    context{ std::exchange(other.context, nullptr) }, queue{ std::exchange(other.queue, nullptr) }, device{ std::exchange(other.device, nullptr) } {}

command_queue &command_queue::operator=(command_queue &&other) {
    if (this != std::addressof(other)) {
        context = std::exchange(other.context, nullptr);
        queue = std::exchange(other.queue, nullptr);
        device = std::exchange(other.device, nullptr);
    }
    return *this;
}

command_queue::~command_queue() {
    if (queue) {
        clReleaseCommandQueue(queue);
    }
}

}  // namespace plssvm::opencl::detail