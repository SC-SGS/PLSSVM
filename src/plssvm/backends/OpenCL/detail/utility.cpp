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
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all, plssvm::detail::to_lower_case, plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::unsupported_kernel_type_exception
#include "plssvm/target_platform.hpp"                       // plssvm::target_platform

#include "CL/cl.h"       // cl_platform_id, cl_device_id, cl_uint, cl_device_type, cl_context, CL_DEVICE_NAME, CL_QUEUE_DEVICE, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
                         // CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, clGetPlatformIDs, clGetDeviceIDs, clGetDeviceInfo, clCreateContext, clCreateCommandQueue, clGetCommandQueueInfo
#include "fmt/format.h"  // fmt::format

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

[[nodiscard]] std::vector<command_queue> get_command_queues(const target_platform target) {
    if (target != target_platform::automatic) {
        return get_command_queues_impl(target);
    } else {
        std::vector<command_queue> target_devices = get_command_queues_impl(target_platform::gpu_nvidia);
        if (target_devices.empty()) {
            target_devices = get_command_queues_impl(target_platform::gpu_amd);
            if (target_devices.empty()) {
                target_devices = get_command_queues_impl(target_platform::gpu_intel);
                if (target_devices.empty()) {
                    target_devices = get_command_queues_impl(target_platform::cpu);
                }
            }
        }
        return target_devices;
    }
}

void device_synchronize(const command_queue &queue) {
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue.queue));
}

[[nodiscard]] std::string get_device_name(const command_queue &queue) {
    // get device
    cl_device_id device_id;
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue.queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");
    // get device name
    std::string device_name(128, '\0');
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name.size() * sizeof(char), device_name.data(), nullptr), "error obtaining device name");
    return device_name.substr(0, device_name.find_first_of('\0'));
}

[[nodiscard]] std::pair<std::string, std::string> kernel_type_to_function_name(const kernel_type kernel) {
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

}  // namespace plssvm::opencl::detail