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

#ifndef PLSSVM_BACKENDS_OPENCL_DETAIL_COMMAND_QUEUE_HPP_
#define PLSSVM_BACKENDS_OPENCL_DETAIL_COMMAND_QUEUE_HPP_
#pragma once

#include "plssvm/backends/OpenCL/detail/kernel.hpp"  // plssvm::opencl::detail::kernel

#include "CL/cl.h"  // cl_context, cl_command_queue, cl_device_id

#include <map>  // std::map

namespace plssvm::opencl::detail {

/**
 * @brief RAII wrapper class around a cl_command_queue.
 * @details Also contains the compiled kernels associated with the command queue.
 */
class command_queue {
  public:
    /**
     * @brief Empty default construct a command queue.
     */
    command_queue() = default;
    /**
     * @brief Construct a command queue with the provided information.
     * @param[in] context the associated OpenCL cl_context
     * @param[in] device the associated OpenCL cl_device_id
     */
    command_queue(cl_context context, cl_device_id device);

    /**
     * @brief Delete copy-constructor to make command_queue a move only type.
     */
    command_queue(const command_queue &) = delete;
    /**
     * @brief Move-constructor as command_queue is a move-only type.
     * @param[in,out] other the command_queue to move the resources from
     */
    command_queue(command_queue &&other) noexcept;
    /**
     * @brief Delete copy-assignment-operator to make command_queue a move only type.
     */
    command_queue &operator=(const command_queue &) = delete;
    /**
     * @brief Move-assignment-operator as command_queue is a move-only type.
     * @param[in,out] other the command_queue to move the resources from
     * @return `*this`
     */
    command_queue &operator=(command_queue &&other) noexcept;

    /**
     * @brief Release the cl_command_queue resources on destruction.
     */
    ~command_queue();

    /**
     * @brief Implicitly convert a command_queue wrapper to an OpenCL cl_command_queue.
     * @return the wrapped OpenCL cl_command_queue (`[[nodiscard]]`)
     */
    [[nodiscard]] operator cl_command_queue &() noexcept { return queue; }
    /**
     * @brief Implicitly convert a command_queue wrapper to an OpenCL cl_command_queue.
     * @return the wrapped OpenCL cl_command_queue (`[[nodiscard]]`)
     */
    [[nodiscard]] operator const cl_command_queue &() const noexcept { return queue; }

    /**
     * @brief Add a new OpenCL @p compute_kernel used for @p name to this command queue.
     * @param[in] name the name of the kernel that is to be added
     * @param[in] compute_kernel the kernel to add
     */
    void add_kernel(compute_kernel_name name, kernel &&compute_kernel);

    /**
     * @brief Get the OpenCL kernel used for @p name.
     * @param[in] name the name of the kernel
     * @throws std::out_of_range if a kernel with @p name is requested that has not been compiled for this command queue
     * @return the compiled kernel (`[[nodiscard]]`)
     */
    [[nodiscard]] const kernel &get_kernel(compute_kernel_name name) const;

    /// The wrapped cl_command_queue.
    cl_command_queue queue{};
    /// All OpenCL device kernel associated with the device corresponding to this command queue.
    std::map<compute_kernel_name, kernel> kernels{};
};

}  // namespace plssvm::opencl::detail

#endif  // PLSSVM_BACKENDS_OPENCL_DETAIL_COMMAND_QUEUE_HPP_