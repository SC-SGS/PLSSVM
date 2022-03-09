/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/error_code.hpp"     // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                             // plssvm::kernel_index_type, plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all, plssvm::detail::to_lower_case, plssvm::detail::contains
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::unsupported_kernel_type_exception, plssvm::invalid_file_format_exception
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "CL/cl.h"        // cl_program, cl_platform_id, cl_device_id, cl_uint, cl_device_type, cl_context, CL_DEVICE_NAME, CL_QUEUE_DEVICE, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
                          // CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_PROGRAM_BUILD_LOG, clCreateProgramWithSource, clBuildProgram, clGetProgramBuildInfo, clCreateKernel, clReleaseProgram,
                          //  clSetKernelArg, clEnqueueNDRangeKernel, clFinish, clGetPlatformIDs, clGetDeviceIDs, clGetDeviceInfo, clCreateContext, clCreateCommandQueue, clGetCommandQueueInfo
#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <fstream>      // std::ifstream
#include <ios>          // std::ios, std::streamsize
#include <limits>       // std::numeric_limits
#include <map>          // std::map
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

namespace plssvm::opencl::detail {

void device_assert(const error_code ec, const std::string_view msg) {
    if (!ec) {
        if (msg.empty()) {
            throw backend_exception{ fmt::format("OpenCL assert ({})!", ec) };
        } else {
            throw backend_exception{ fmt::format("OpenCL assert ({}): {}!", ec, msg) };
        }
    }
}

[[nodiscard]] std::vector<command_queue> get_command_queues_impl(const target_platform target) {
    std::map<cl_platform_id, std::vector<cl_device_id>> platform_devices;

    // get number of platforms
    cl_uint num_platforms;
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms), "error retrieving the number of available platforms");
    // get platforms
    std::vector<cl_platform_id> platform_ids(num_platforms);
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr), "error retrieving platform IDs");

    for (const cl_platform_id &platform : platform_ids) {
        // get number of devices
        cl_uint num_devices;
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices), "error retrieving the number of devices");
        // get devices
        std::vector<cl_device_id> device_ids(num_devices);
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr), "error retrieving device IDs");

        for (const cl_device_id &device : device_ids) {
            cl_device_type device_type;
            PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr), "error retrieving the device type");
            if (target == target_platform::cpu && device_type == CL_DEVICE_TYPE_CPU) {
                // select CPU device
                platform_devices[platform].push_back(device);
            } else {
                // must be a GPU device
                if (device_type == CL_DEVICE_TYPE_GPU) {
                    // get vendor name of current GPU
                    std::string vendor_string(128, '\0');
                    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendor_string.size() * sizeof(char), vendor_string.data(), nullptr), "error retrieving device information");
                    vendor_string = vendor_string.substr(0, vendor_string.find_first_of('\0'));
                    // convert vendor name to lower case
                    ::plssvm::detail::to_lower_case(vendor_string);

                    switch (target) {
                        case target_platform::gpu_nvidia:
                            if (::plssvm::detail::contains(vendor_string, "nvidia")) {
                                platform_devices[platform].push_back(device);
                            }
                            break;
                        case target_platform::gpu_amd:
                            if (::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices")) {
                                platform_devices[platform].push_back(device);
                            }
                            break;
                        case target_platform::gpu_intel:
                            if (::plssvm::detail::contains(vendor_string, "intel")) {
                                platform_devices[platform].push_back(device);
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }

    std::vector<command_queue> command_queues;
    for (const auto &[platform, device_list] : platform_devices) {
        error_code err;
        cl_context context = clCreateContext(nullptr, static_cast<cl_uint>(device_list.size()), device_list.data(), nullptr, nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL context");

        for (const cl_device_id &device : device_list) {
#ifdef CL_VERSION_2_0
            // use new clCreateCommandQueueWithProperties function
            command_queues.emplace_back(context, clCreateCommandQueueWithProperties(context, device, 0, &err), device);
#else
            // use old clCreateCommandQueue function (deprecated in newer OpenCL versions)
            command_queues.emplace_back(context, clCreateCommandQueue(context, device, 0, &err), device);
#endif
            PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL command queue");
        }
    }

    return command_queues;
}

std::pair<std::vector<command_queue>, target_platform> get_command_queues(const target_platform target) {
    if (target != target_platform::automatic) {
        return std::make_pair(get_command_queues_impl(target), target);
    } else {
        target_platform used_target = target_platform::gpu_nvidia;
        std::vector<command_queue> target_devices = get_command_queues_impl(used_target);
        if (target_devices.empty()) {
            used_target = target_platform::gpu_amd;
            target_devices = get_command_queues_impl(used_target);
            if (target_devices.empty()) {
                used_target = target_platform::gpu_intel;
                target_devices = get_command_queues_impl(used_target);
                if (target_devices.empty()) {
                    used_target = target_platform::cpu;
                    target_devices = get_command_queues_impl(used_target);
                }
            }
        }
        return std::make_pair(std::move(target_devices), used_target);
    }
}

void device_synchronize(const command_queue &queue) {
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue.queue));
}

std::string get_device_name(const command_queue &queue) {
    // get device
    cl_device_id device_id;
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue.queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");
    // get device name
    std::string device_name(128, '\0');
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name.size() * sizeof(char), device_name.data(), nullptr), "error obtaining device name");
    return device_name.substr(0, device_name.find_first_of('\0'));
}

std::pair<std::string, std::string> kernel_type_to_function_name(const kernel_type kernel) {
    switch (kernel) {
        case kernel_type::linear:
            return std::make_pair("device_kernel_q_linear", "device_kernel_linear");
        case kernel_type::polynomial:
            return std::make_pair("device_kernel_q_poly", "device_kernel_poly");
        case kernel_type::rbf:
            return std::make_pair("device_kernel_q_radial", "device_kernel_radial");
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", ::plssvm::detail::to_underlying(kernel)) };
}

template <typename real_type>
std::vector<kernel> create_kernel(const std::vector<command_queue> &queues, const std::string &file, const std::string &kernel_name) {
    std::string kernel_src_string;

    // append kernel file to kernel string
    const auto append_to_kernel_src_string = [&kernel_src_string](const std::string &file_name) {
        std::ifstream in{ file_name };

        PLSSVM_ASSERT(in.good(), fmt::format("couldn't open kernel source file ({})", file_name));

        in.ignore(std::numeric_limits<std::streamsize>::max());
        std::streamsize len = in.gcount();
        in.clear();
        in.seekg(0, std::ios::beg);

        PLSSVM_ASSERT(len > 0, fmt::format("empty file ({})", file_name));

        const std::string::size_type old_size = kernel_src_string.size();
        kernel_src_string.resize(old_size + len);
        if (!in.read(kernel_src_string.data() + old_size, len)) {
            throw invalid_file_format_exception{ fmt::format("Error while reading file: '{}'!", file_name) };
        }
    };

    // read atomic
    append_to_kernel_src_string(PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "detail/atomics.cl");
    // read kernel
    append_to_kernel_src_string(file);

    // replace type
    ::plssvm::detail::replace_all(kernel_src_string, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
    ::plssvm::detail::replace_all(kernel_src_string, "kernel_index_type", ::plssvm::detail::arithmetic_type_name<::plssvm::kernel_index_type>());
    // replace constants
    ::plssvm::detail::replace_all(kernel_src_string, "INTERNAL_BLOCK_SIZE", fmt::format("{}", INTERNAL_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "THREAD_BLOCK_SIZE", fmt::format("{}", THREAD_BLOCK_SIZE));

    error_code err;

    // create program
    const char *kernel_src_ptr = kernel_src_string.c_str();
    // TODO: not all command queue must have the same context (but this would be highly unlikely)
    cl_program program = clCreateProgramWithSource(queues[0].context, 1, &kernel_src_ptr, nullptr, &err);
    err = clBuildProgram(program, 0, nullptr, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
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

template std::vector<kernel> create_kernel<float>(const std::vector<command_queue> &, const std::string &, const std::string &);
template std::vector<kernel> create_kernel<double>(const std::vector<command_queue> &, const std::string &, const std::string &);

}  // namespace plssvm::opencl::detail