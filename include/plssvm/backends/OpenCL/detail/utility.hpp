/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Utility functions specific to the OpenCL backend.
 */

#pragma once

#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/exceptions.hpp"         // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                          // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all

#include "CL/cl.h"     // cl_program, clCreateProgramWithSource, clBuildProgram, clGetProgramBuildInfo, cl_kernel, clCreateKernel
                       // clReleaseProgram, cl_uint, clSetKernelArg
#include "fmt/core.h"  // fmt::format

#include <cstddef>  // std::size_t
#include <fstream>  // std::ifstream
#include <ios>      // std::ios, std::streamsize
#include <string>   // std::string
#include <utility>  // std::forward
#include <vector>   // std::vector

namespace plssvm::opencl::detail {

std::string get_device_name(cl_command_queue queue) {
    error_code err;
    // get device
    cl_device_id device_id;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr);
    if (!err) {
        throw backend_exception{ fmt::format("Error obtaining device ({})!", err) };
    }
    // get device name
    std::string device_name(128, '\0');
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name.size() * sizeof(char), device_name.data(), nullptr);
    if (!err) {
        throw backend_exception{ fmt::format("Error obtaining device name ({})!", err) };
    }
    return device_name.substr(0, device_name.find_first_of('\0'));
}

/**
 * @brief Create a kernel with @p kernel_name for the given devices and context from the file @p file.
 * @tparam real_type the floating point type used to replace the placeholders in the kernel file
 * @tparam size_type the unsigned integer type used to replace the placeholders in the kernel file
 * @param[in] context the current OpenCL context
 * @param[in] device_id the current OpenCL device
 * @param[in] file the file containing the kernel
 * @param[in] kernel_name the name of the kernel to create
 * @return the kernel
 */
template <typename real_type, typename size_type = std::size_t>
cl_kernel create_kernel(cl_command_queue queue, const std::string &file, const std::string &kernel_name) {
    // read kernel
    std::string kernel_src_string;
    {
        std::ifstream in{ file };
        if (in.fail()) {
            throw backend_exception{ fmt::format("Couldn't open kernel source file: {}!", file) };
        }
        in.seekg(0, std::ios::end);
        std::streamsize len = in.tellg();
        in.seekg(0, std::ios::beg);

        kernel_src_string.resize(len);
        in.read(kernel_src_string.data(), len);
    }

    // replace type
    ::plssvm::detail::replace_all(kernel_src_string, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
    ::plssvm::detail::replace_all(kernel_src_string, "size_type", ::plssvm::detail::arithmetic_type_name<size_type>());
    // replace constants
    ::plssvm::detail::replace_all(kernel_src_string, "INTERNAL_BLOCK_SIZE", fmt::format("{}", INTERNAL_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "THREAD_BLOCK_SIZE", fmt::format("{}", THREAD_BLOCK_SIZE));

    error_code err;

    // get context
    cl_context context;  // TODO: RAII
    err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr);
    if (!err) {
        throw backend_exception{ fmt::format("Error obtaining context ({})!", err) };
    }
    // get device
    cl_device_id device;
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, nullptr);
    if (!err) {
        throw backend_exception{ fmt::format("Error obtaining device ({})!", err) };
    }

    // create program
    const char *kernel_src_ptr = kernel_src_string.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_src_ptr, nullptr, &err);
    if (!err) {
        throw backend_exception{ fmt::format("Error creating OpenCL program ({})!", err) };
    }

    err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (!err) {
        // collect build log
        std::size_t len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
        std::string buffer(len, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, buffer.data(), nullptr);
        buffer = buffer.substr(0, buffer.find_first_of('\0'));
        throw backend_exception{ fmt::format("Error building OpenCL program ({})!:\n{}", err, buffer) };
    }

    // create kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (!err) {
        throw backend_exception{ fmt::format("Error creating OpenCL kernel ({})!", err) };
    }

    // release resource
    if (program) {
        err = clReleaseProgram(program);
        if (!err) {
            throw backend_exception{ fmt::format("Error releasing OpenCL program resources ({})!", err) };
        }
    }

    return kernel;
}

/**
 * @brief Set all arguments in the parameter pack @p args for the kernel @p kernel.
 * @tparam Args the types of the arguments
 * @param[in] kernel the OpenCL kernel to set the arguments
 * @param[in] args the arguments to set
 */
template <typename... Args>
void set_kernel_args(cl_kernel kernel, Args... args) {
    cl_uint i = 0;
    // iterate over parameter pack and set OpenCL kernel
    ([&](auto &arg) {
        error_code err = clSetKernelArg(kernel, i++, sizeof(decltype(arg)), &arg);
        if (!err) {
            throw backend_exception{ fmt::format("Error setting OpenCL kernel argument {} ({})!", i - 1, err) };
        }
    }(args),
     ...);
}

/**
 * @brief
 * @tparam Args the types of the arguments
 * @param[inout] queue the command queue on which the kernel should be executed
 * @param[inout] kernel the kernel to run
 * @param[in] grid_size the number of global work-items (possibly multi-dimensional)
 * @param[in] block_size the number of work-items that make up a work-group (possibly multi-dimensional)
 * @param[in] args the arguments to set
 */
template <typename... Args>
void run_kernel(cl_command_queue queue, cl_kernel kernel, std::vector<std::size_t> grid_size, std::vector<std::size_t> block_size, Args &&...args) {
    PLSSVM_ASSERT(grid_size.size() == block_size.size(), "grid_size and block_size must have the same number of dimensions!: {} != {}", grid_size.size(), block_size.size());
    PLSSVM_ASSERT(grid_size.size() <= 3, "The number of dimensions must be less or equal than 3!: {} > 3", grid_size.size());

    // set kernel arguments
    set_kernel_args(kernel, std::forward<Args>(args)...);

    // enqueue kernel in command queue
    error_code err = clEnqueueNDRangeKernel(queue, kernel, grid_size.size(), nullptr, grid_size.data(), block_size.data(), 0, nullptr, nullptr);
    if (!err) {
        throw backend_exception{ fmt::format("Error enqueuing OpenCL kernel ({})!", err) };
    }

    // wait until kernel computation finished
    err = clFinish(queue);
    if (!err) {
        throw backend_exception{ fmt::format("Error running OpenCL kernel ({})!", err) };
    }
}

/**
 * @brief
 * @tparam Args the types of the arguments
 * @param[inout] queue the command queue on which the kernel should be executed
 * @param[inout] kernel the kernel to run
 * @param[in] grid_size the number of global work-items
 * @param[in] block_size the number of work-items that make up a work-group
 * @param[in] args the arguments to set
 */
template <typename... Args>
void run_kernel(cl_command_queue queue, cl_kernel kernel, std::size_t grid_size, std::size_t block_size, Args &&...args) {
    run_kernel(queue, kernel, std::vector<std::size_t>{ grid_size }, std::vector<std::size_t>{ block_size }, std::forward<Args>(args)...);
}

}  // namespace plssvm::opencl::detail