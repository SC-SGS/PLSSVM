/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions specific to the OpenCL backend.
 */

#pragma once

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/error_code.hpp"     // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/kernel.hpp"         // plssvm::opencl::detail::kernel
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "CL/cl.h"  // cl_kernel, cl_uint, cl_int

#include "fmt/core.h"  // fmt::format

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::forward, std::pair
#include <vector>       // std::vector

/**
 * @def PLSSVM_OPENCL_ERROR_CHECK
 * @brief Macro used for error checking OpenCL runtime functions.
 */
#define PLSSVM_OPENCL_ERROR_CHECK(err, ...) plssvm::opencl::detail::device_assert((err), ##__VA_ARGS__)

namespace plssvm::opencl::detail {

/**
 * @brief Check the OpenCL error @p code. If @p code signals an error, throw a `plssvm::opencl::backend_exception`.
 * @details The exception contains the error name and additional debug information.
 * @param[in] code the OpenCL error code to check
 * @param[in] msg optional message printed if the error code check failed
 * @throws `plssvm::opencl::backend_exception` if the error code signals a failure
 */
void device_assert(error_code code, std::string_view msg = "");

/**
 * @brief Returns the list devices matching the target platform @p target.
 * @details If the selected target platform is `plssvm::target_platform::automatic` the selector tries to find devices in the following order:
 *          1. NVIDIA GPUs
 *          2. AMD GPUs
 *          3. Intel GPUs
 *          4. CPUs
 *
 * @param[in] target the target platform for which the devices must match
 * @return the command queues (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<command_queue> get_command_queues(target_platform target);

/**
 * @brief Wait for the compute device associated with @p queue to finish.
 * @param[in] queue the command queue to synchronize
 */
void device_synchronize(const command_queue &queue);

/**
 * @brief Get the name of the device associated with the OpenCL command queue @p queue.
 * @param[in] queue the OpenCL command queue
 * @return the device name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_device_name(const command_queue &queue);

/**
 * @brief Convert the kernel type @p kernel to the function names for the q and svm kernel functions.
 * @param[in] kernel the kernel type
 * @return the kernel function names (first: q_kernel name, second: svm_kernel name) (`[[nodiscard]]`)
 */
[[nodiscard]] std::pair<std::string, std::string> kernel_type_to_function_name(kernel_type kernel);

/**
 * @brief Create a kernel with @p kernel_name for the given command queues from the file @p file.
 * @tparam real_type the floating point type used to replace the placeholders in the kernel file
 * @param[in] queues the used OpenCL command queues
 * @param[in] file the file containing the kernel
 * @param[in] kernel_name the name of the kernel to create
 * @throws plssvm::invalid_file_format_exception if the file couldn't be read using [`std::ifstream::read`](https://en.cppreference.com/w/cpp/io/basic_istream/read)
 * @return the kernel (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] std::vector<kernel> create_kernel(const std::vector<command_queue> &queues, const std::string &file, const std::string &kernel_name);

/**
 * @brief Set all arguments in the parameter pack @p args for the kernel @p kernel.
 * @tparam Args the types of the arguments
 * @param[in] kernel the OpenCL kernel to set the arguments
 * @param[in] args the arguments to set
 */
template <typename... Args>
inline void set_kernel_args(cl_kernel kernel, Args... args) {
    cl_uint i = 0;
    // iterate over parameter pack and set OpenCL kernel
    ([&](auto &arg) {
        const error_code ec = clSetKernelArg(kernel, i++, sizeof(decltype(arg)), &arg);
        PLSSVM_OPENCL_ERROR_CHECK(ec, fmt::format("error setting OpenCL kernel argument {}", i - 1));
    }(args),
     ...);
}

/**
 * @brief Run the 1D @p kernel on the @p queue with the additional parameters @p args.
 * @tparam Args the types of the arguments
 * @param[in,out] queue the command queue on which the kernel should be executed
 * @param[in,out] kernel the kernel to run
 * @param[in] grid_size the number of global work-items (possibly multi-dimensional)
 * @param[in] block_size the number of work-items that make up a work-group (possibly multi-dimensional)
 * @param[in] args the arguments to set
 */
template <typename... Args>
inline void run_kernel(const command_queue &queue, cl_kernel kernel, const std::vector<std::size_t> &grid_size, const std::vector<std::size_t> &block_size, Args &&...args) {
    PLSSVM_ASSERT(grid_size.size() == block_size.size(), "grid_size and block_size must have the same number of dimensions!: {} != {}", grid_size.size(), block_size.size());
    PLSSVM_ASSERT(grid_size.size() <= 3, "The number of dimensions must be less or equal than 3!: {} > 3", grid_size.size());

    // set kernel arguments
    set_kernel_args(kernel, std::forward<Args>(args)...);

    // enqueue kernel in command queue
    PLSSVM_OPENCL_ERROR_CHECK(clEnqueueNDRangeKernel(queue.queue, kernel, static_cast<cl_int>(grid_size.size()), nullptr, grid_size.data(), block_size.data(), 0, nullptr, nullptr), "error enqueuing OpenCL kernel");
    // wait until kernel computation finished
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue.queue), "error running OpenCL kernel");
}

/**
 * @brief Run the 1D @p kernel on the @p queue with the additional parameters @p args.
 * @tparam Args the types of the arguments
 * @param[in,out] queue the command queue on which the kernel should be executed
 * @param[in,out] kernel the kernel to run
 * @param[in] grid_size the number of global work-items
 * @param[in] block_size the number of work-items that make up a work-group
 * @param[in] args the arguments to set
 */
template <typename... Args>
inline void run_kernel(const command_queue &queue, cl_kernel kernel, std::size_t grid_size, std::size_t block_size, Args &&...args) {
    run_kernel(queue, kernel, std::vector<std::size_t>{ grid_size }, std::vector<std::size_t>{ block_size }, std::forward<Args>(args)...);
}

}  // namespace plssvm::opencl::detail
