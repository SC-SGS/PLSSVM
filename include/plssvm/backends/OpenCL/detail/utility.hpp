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
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // PLSSVM_OPENCL_ERROR_CHECK
#include "plssvm/constants.hpp"                             // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platform.hpp"                       // plssvm::target_platform

#include "CL/cl.h"  // cl_program, cl_kernel, cl_uint, cl_int, CL_PROGRAM_BUILD_LOG, clCreateProgramWithSource, clBuildProgram, clGetProgramBuildInfo,
                    // clCreateKernel, clReleaseProgram, clSetKernelArg, clEnqueueNDRangeKernel, clFinish

#include "fmt/core.h"  // fmt::format

#include <cstddef>  // std::size_t
#include <fstream>  // std::ifstream
#include <ios>      // std::ios, std::streamsize
#include <limits>   // std::numeric_limits
#include <string>   // std::string
#include <utility>  // std::forward, std::pair
#include <vector>   // std::vector

#define PLSSVM_OPENCL_ERROR_CHECK(err, ...) plssvm::opencl::detail::device_assert((err), ##__VA_ARGS__)

namespace plssvm::opencl::detail {

/**
 * @brief Check the OpenCL error @p code. If @p code signals an error, throw a `plssvm::opencl::backend_exception`.
 * @details The exception contains the error name and additional debug information.
 * @param[in] code the OpenCL error code to check
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
 * @return the device name
 */
[[nodiscard]] std::string get_device_name(const command_queue &queue);

/**
 * @brief Convert the kernel type @p kernel to the function names for the q and svm kernel functions.
 * @param[in] kernel the kernel type
 * @return the kernel function names (first: q_kernel name, second: svm_kernel name)
 */
[[nodiscard]] std::pair<std::string, std::string> kernel_type_to_function_name(kernel_type kernel);

/**
 * @brief Create a kernel with @p kernel_name for the given command queues from the file @p file.
 * @tparam real_type the floating point type used to replace the placeholders in the kernel file
 * @tparam kernel_index_type the unsigned integer type used to replace the placeholders in the kernel file
 * @param[in] queues the used OpenCL command queues
 * @param[in] file the file containing the kernel
 * @param[in] kernel_name the name of the kernel to create
 * @return the kernel
 */
template <typename real_type, typename kernel_index_type>
[[nodiscard]] inline std::vector<kernel> create_kernel(const std::vector<command_queue> &queues, const std::string &file, const std::string &kernel_name) {
    // read kernel
    std::string kernel_src_string;
    {
        std::ifstream in{ file };

        PLSSVM_ASSERT(in.good(), fmt::format("couldn't open kernel source file ({})", file));

        in.ignore(std::numeric_limits<std::streamsize>::max());
        std::streamsize len = in.gcount();
        in.clear();
        in.seekg(0, std::ios::beg);

        PLSSVM_ASSERT(len > 0, fmt::format("empty file ({})", file));

        kernel_src_string.resize(len);
        in.read(kernel_src_string.data(), len);
    }

    // replace type
    ::plssvm::detail::replace_all(kernel_src_string, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
    ::plssvm::detail::replace_all(kernel_src_string, "kernel_index_type", ::plssvm::detail::arithmetic_type_name<kernel_index_type>());
    // replace constants
    ::plssvm::detail::replace_all(kernel_src_string, "INTERNAL_BLOCK_SIZE", fmt::format("{}", INTERNAL_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "THREAD_BLOCK_SIZE", fmt::format("{}", THREAD_BLOCK_SIZE));

    error_code err;

    // create program
    const char *kernel_src_ptr = kernel_src_string.c_str();
    // TODO: not all command queue must have the same context (but this would be highly unlikely)
    cl_program program = clCreateProgramWithSource(queues[0].context, 1, &kernel_src_ptr, nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error creating OpenCL program");
    err = clBuildProgram(program, 0, nullptr, "-I " PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY " -cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
    if (!err) {
        // determine the size of the log
        std::size_t log_size;
        clGetProgramBuildInfo(program, queues[0].device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        // allocate memory for the log
        std::string log(log_size, ' ');
        // get the log
        clGetProgramBuildInfo(program, queues[0].device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        // print the log
        PLSSVM_OPENCL_ERROR_CHECK(err, fmt::format("error building OpenCL program ({})", log));
    }

    // build kernels
    std::vector<kernel> kernels;
    for ([[maybe_unused]] const command_queue &q : queues) {
        // create kernel
        kernels.emplace_back(clCreateKernel(program, kernel_name.c_str(), &err));
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating OpenCL kernel");
    }

    // release resource
    if (program) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseProgram(program), "error releasing OpenCL program resources");
    }

    return kernels;
}

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
