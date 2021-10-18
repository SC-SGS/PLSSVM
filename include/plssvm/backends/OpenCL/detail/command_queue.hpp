/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a very small RAII wrapper around a cl_command_queue including information about its associated OpenCL context and device.
 */

#pragma once

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id, clReleaseCommandQueue

#include <memory>   // std::addressof
#include <utility>  // std::exchange

namespace plssvm::opencl::detail {

/**
 * @brief RAII wrapper class around a cl_command_queue.
 * @details Also contains information about the associated cl_context and cl_device_id.
 */
class command_queue {
  public:
    /**
     * @brief Empty default construct command queue.
     */
    command_queue() = default;
    /**
     * @brief Construct a command queue with the provided information.
     * @param[in] p_context the associated OpenCL cl_context
     * @param[in] p_queue the OpenCL cl_command_queue to wrap
     * @param[in] p_device the associated OpenCL cl_device_id
     */
    command_queue(cl_context p_context, cl_command_queue p_queue, cl_device_id p_device) :
        context{ p_context }, queue{ p_queue }, device{ p_device } {}

    /**
     * @brief Delete copy-constructor to make `command_queue` a move only type.
     */
    command_queue(const command_queue &) = delete;
    /**
     * @brief Move-constructor as `command_queue` is a move-only type.
     * @param[in,out] other the command_queue to move the resources from
     */
    command_queue(command_queue &&other) noexcept :
        context{ std::exchange(other.context, nullptr) }, queue{ std::exchange(other.queue, nullptr) }, device{ std::exchange(other.device, nullptr) } {}
    /**
     * @brief Delete copy-assignment-operator to make `command_queue` a move only type.
     */
    command_queue &operator=(const command_queue &) = delete;
    /**
     * @brief Move-assignment-operator as `command_queue` is a move-only type-
     * @param[in,out] other the command_queue to move the resources from
     * @return `*this`
     */
    command_queue &operator=(command_queue &&other) {
        if (this != std::addressof(other)) {
            context = std::exchange(other.context, nullptr);
            queue = std::exchange(other.queue, nullptr);
            device = std::exchange(other.device, nullptr);
        }
        return *this;
    }

    /**
     * @brief Release the cl_command_queue resources on destruction.
     */
    ~command_queue() {
        if (queue) {
            clReleaseCommandQueue(queue);
        }
    }

    /// The OpenCL context associated with the wrapped cl_command_queue.
    cl_context context{};
    /// The wrapped cl_command_queue.
    cl_command_queue queue{};
    /// The OpenCL device associated with the wrapped cl_command_queue.
    cl_device_id device{};
};

}  // namespace plssvm::opencl::detail