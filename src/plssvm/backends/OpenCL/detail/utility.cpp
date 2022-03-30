/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "plssvm/backends/OpenCL/detail/context.hpp"     // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/exceptions.hpp"         // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                          // plssvm::kernel_index_type, plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all, plssvm::detail::to_lower_case, plssvm::detail::contains
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all
#include "plssvm/exceptions/exceptions.hpp"              // plssvm::unsupported_kernel_type_exception, plssvm::invalid_file_format_exception
#include "plssvm/target_platforms.hpp"                   // plssvm::target_platform

#include "CL/cl.h"        // cl_program, cl_platform_id, cl_device_id, cl_uint, cl_device_type, cl_context, CL_DEVICE_NAME, CL_QUEUE_DEVICE, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU,
                          // CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_PROGRAM_BUILD_LOG, clCreateProgramWithSource, clBuildProgram, clGetProgramBuildInfo, clCreateKernel, clReleaseProgram,
                          //  clSetKernelArg, clEnqueueNDRangeKernel, clFinish, clGetPlatformIDs, clGetDeviceIDs, clGetDeviceInfo, clCreateContext, clCreateCommandQueue, clGetCommandQueueInfo
#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <algorithm>
#include <filesystem>  // std::filesystem::exists
#include <fstream>     // std::ifstream
#include <ios>         // std::ios, std::streamsize
#include <limits>      // std::numeric_limits
#include <map>         // std::map
#include <regex>
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

#include <iostream>
#include <string.h>

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

[[nodiscard]] std::vector<context> get_command_queues_impl(const target_platform target) {
    error_code err;

    const auto add_to_map = [](auto &map, const auto &key, auto value) {
        if (map.count(key) == 0) {
            map[key] = std::vector<decltype(value)>();
        }
        map[key].push_back(value);
    };

    // iterate over all platforms and save all available devices
    std::map<std::pair<cl_platform_id, target_platform>, std::vector<cl_device_id>> platform_devices;
    // get number of platforms
    cl_uint num_platforms;
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms), "error retrieving the number of available platforms");
    // get platforms
    std::vector<cl_platform_id> platform_ids(num_platforms);
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr), "error retrieving platform IDs");

    for (const cl_platform_id &platform : platform_ids) {
        // retrieve platform name
        //        std::size_t platform_name_size;
        //        PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platform_name_size), "error retrieving the size of the platform name");
        //        std::string platform_name(platform_name_size, '\0');
        //        PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name.data(), nullptr), "error retrieving the platform name");

        // get devices associated with current platform
        cl_uint num_devices;
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices), "error retrieving the number of devices");
        // get devices
        std::vector<cl_device_id> device_ids(num_devices);
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr), "error retrieving device IDs");

        for (const cl_device_id &device : device_ids) {
            // get device type
            cl_device_type device_type;
            PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr), "error retrieving the device type");

            if (device_type == CL_DEVICE_TYPE_CPU) {
                // is CPU device
#if defined(PLSSVM_HAS_CPU_TARGET)
                add_to_map(platform_devices, std::make_pair(platform, target_platform::cpu), device);
#endif
            } else if (device_type == CL_DEVICE_TYPE_GPU) {
                // is GPU device TODO
                std::string vendor_string(128, '\0');
                PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendor_string.size() * sizeof(char), vendor_string.data(), nullptr), "error retrieving device information");
                vendor_string = vendor_string.substr(0, vendor_string.find_first_of('\0'));
                // convert vendor name to lower case
                ::plssvm::detail::to_lower_case(vendor_string);

                //                fmt::print("vendor string: {}\n", vendor_string);

                if (::plssvm::detail::contains(vendor_string, "nvidia")) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_nvidia), device);
#endif
                } else if (::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices")) {
#if defined(PLSSVM_HAS_AMD_TARGET)
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_amd), device);
#endif
                } else if (::plssvm::detail::contains(vendor_string, "intel")) {
#if defined(PLSSVM_HAS_INTEL_TARGET)
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_intel), device);
#endif
                }
            }
        }

        // create context associated with platform and devices
        //        cl_context context = clCreateContext(nullptr, static_cast<cl_uint>(device_list.size()), device_list.data(), nullptr, nullptr, &err);
        //        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL context");
    }

    // only interested in devices on the current target platform
    const auto erase_if = [](auto &c, const auto &pred) {
        auto old_size = c.size();
        for (auto i = c.begin(), last = c.end(); i != last;) {
            if (pred(*i)) {
                i = c.erase(i);
            } else {
                ++i;
            }
        }
        return old_size - c.size();
    };
    erase_if(platform_devices, [target](const auto &item) {
        const auto &[key, value] = item;
        return key.second != target;
    });

    std::vector<context> contexts;
    for (auto &[platform, devices] : platform_devices) {
        // TODO: platform in context
        cl_context cont = clCreateContext(nullptr, static_cast<cl_uint>(devices.size()), devices.data(), nullptr, nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL context");
        contexts.emplace_back(cont, std::move(devices));
    }

    //    for (const auto &cont : contexts) {
    //        std::cout << cont << std::endl;
    //    }
    return contexts;
}

std::pair<std::vector<context>, target_platform> get_command_queues(const target_platform target) {
    if (target != target_platform::automatic) {
        return std::make_pair(get_command_queues_impl(target), target);
    } else {
        target_platform used_target = target_platform::gpu_nvidia;
        std::vector<context> target_devices = get_command_queues_impl(used_target);
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

void device_synchronize(const cl_command_queue &queue) {
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue));
}

std::string get_device_name(const cl_command_queue &queue) {
    // get device
    cl_device_id device_id;
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");
    // get device name
    std::string device_name(128, '\0');  // TODO:
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, device_name.size() * sizeof(char), device_name.data(), nullptr), "error obtaining device name");
    return device_name.substr(0, device_name.find_first_of('\0'));
}

std::vector<std::string> kernel_type_to_function_names(const kernel_type kernel) {
    switch (kernel) {
        case kernel_type::linear:
            return { "device_kernel_q_linear", "device_kernel_linear", "device_kernel_w_linear" };
        case kernel_type::polynomial:
            return { "device_kernel_q_poly", "device_kernel_poly", "device_kernel_predict_poly" };
        case kernel_type::rbf:
            return { "device_kernel_q_radial", "device_kernel_radial", "device_kernel_predict_radial" };
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", ::plssvm::detail::to_underlying(kernel)) };
}

template <typename real_type>
std::vector<std::vector<kernel>> create_kernel(const std::vector<context> &contexts, const std::vector<std::string> &kernel_sources, const std::vector<std::string> &kernel_names) {
    // TODO: allow more than one context!
    PLSSVM_ASSERT(contexts.size() == 1, fmt::format("currently only a single context is allowed but {} were provided!", contexts.size()));

    error_code err, err_bin;

    std::vector<std::size_t> binary_sizes(contexts[0].devices.size());
    std::vector<unsigned char *> binaries(contexts[0].devices.size());

    const std::string cache_dir_name{ "opencl_cache" };
    std::size_t fileCount = 0;

    if (std::filesystem::exists(cache_dir_name)) {
        auto dirIter = std::filesystem::directory_iterator(cache_dir_name);

        fileCount = std::count_if(
            begin(dirIter),
            end(dirIter),
            [](auto &entry) { return entry.is_regular_file(); });
    }

    if (!std::filesystem::exists(cache_dir_name) && fileCount != contexts[0].devices.size()) {
        fmt::print("Building OpenCL kernels from source.\n");
        // read kernel source files
        std::vector<std::string> kernel_src_strings;
        kernel_src_strings.reserve(kernel_sources.size());
        for (const std::string &file_name : kernel_sources) {
            // read file
            std::ifstream in{ fmt::format("{}{}", PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY, file_name) };
            PLSSVM_ASSERT(in.good(), fmt::format("couldn't open kernel source file ({}{})", PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY, file_name));
            std::string &source = kernel_src_strings.emplace_back((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

            // replace types
            ::plssvm::detail::replace_all(source, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
            ::plssvm::detail::replace_all(source, "kernel_index_type", ::plssvm::detail::arithmetic_type_name<::plssvm::kernel_index_type>());
            // replace constants
            ::plssvm::detail::replace_all(source, "INTERNAL_BLOCK_SIZE", fmt::format("{}", INTERNAL_BLOCK_SIZE));
            ::plssvm::detail::replace_all(source, "THREAD_BLOCK_SIZE", fmt::format("{}", THREAD_BLOCK_SIZE));
        }

        // convert strings to const char*
        std::vector<const char *> kernel_src_ptrs(kernel_src_strings.size());
        for (std::size_t i = 0; i < kernel_src_ptrs.size(); ++i) {
            kernel_src_ptrs[i] = kernel_src_strings[i].c_str();
        }

        // create and build program
        cl_program program = clCreateProgramWithSource(contexts[0].device_context, static_cast<cl_uint>(kernel_src_ptrs.size()), kernel_src_ptrs.data(), nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating program from source");
        err = clBuildProgram(program, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        if (!err) {
            // determine the size of the log
            std::size_t log_size;
            clGetProgramBuildInfo(program, contexts[0].devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            // allocate memory for the log
            std::string log(log_size, ' ');
            // get the log
            clGetProgramBuildInfo(program, contexts[0].devices[0], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            // print the log
            PLSSVM_OPENCL_ERROR_CHECK(err, fmt::format("error building OpenCL program ({})", log));
        }

        // get sizes of binaries
        err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, contexts[0].devices.size() * sizeof(std::size_t), binary_sizes.data(), nullptr);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error retrieving the kernel (binary) kernel sizes");
        for (std::size_t i = 0; i < binaries.size(); ++i) {
            binaries[i] = new unsigned char[binary_sizes[i]];
        }

        // get binaries
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, contexts[0].devices.size() * sizeof(unsigned char *), binaries.data(), nullptr);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error retrieving the kernel binaries");
        // write binaries to file
        for (std::size_t i = 0; i < binary_sizes.size(); ++i) {
            std::filesystem::path p{ cache_dir_name };
            std::filesystem::create_directories(p);
            std::ofstream out{ p / fmt::format("device_{}.bin", i) };
            PLSSVM_ASSERT(out.good(), fmt::format("couldn't create binary cache file (\"{}\") for device {}", p / fmt::format("device_{}.bin", i), i));
            out.write(reinterpret_cast<char *>(binaries[i]), binary_sizes[i]);
        }

        // release resource
        if (program) {
            PLSSVM_OPENCL_ERROR_CHECK(clReleaseProgram(program), "error releasing OpenCL program resources");
        }
    } else {
        fmt::print("Using cached OpenCL kernel binaries.\n");
        const auto common_read_file = [](const char *path, std::size_t *length_out) -> unsigned char * {
            char *buffer;
            FILE *f;
            long length;

            f = fopen(path, "r");
            fseek(f, 0, SEEK_END);
            length = ftell(f);
            fseek(f, 0, SEEK_SET);
            buffer = new char[length];
            if (fread(buffer, 1, length, f) < (size_t) length) {
                return NULL;
            }
            fclose(f);
            if (NULL != length_out) {
                *length_out = length;
            }
            return (unsigned char *) buffer;
        };

        auto dirIter = std::filesystem::directory_iterator(cache_dir_name);
        for (const std::filesystem::directory_entry &entry : dirIter) {
            if (entry.path().string().empty()) {
                continue;
            }
            int i = std::stoi(std::regex_replace(entry.path().string(), std::regex("[^0-9]+"), std::string("$1")));
            binaries[i] = common_read_file(entry.path().c_str(), &binary_sizes[i]);
        }
    }

    // build from binaries
    cl_program binary_program = clCreateProgramWithBinary(contexts[0].device_context, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), binary_sizes.data(), const_cast<const unsigned char **>(binaries.data()), &err_bin, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err_bin, "error loading binaries");
    PLSSVM_OPENCL_ERROR_CHECK(err, "error creating binary program");
    err = clBuildProgram(binary_program, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), nullptr, nullptr, nullptr);
    if (!err) {
        // determine the size of the log
        std::size_t log_size;
        clGetProgramBuildInfo(binary_program, contexts[0].devices[0], CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        // allocate memory for the log
        std::string log(log_size, ' ');
        // get the log
        clGetProgramBuildInfo(binary_program, contexts[0].devices[0], CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        // print the log
        PLSSVM_OPENCL_ERROR_CHECK(err, fmt::format("error building OpenCL binary program ({})", log));
    }

    // build all kernels, one for each device
    std::vector<std::vector<kernel>> kernels(kernel_names.size());
    for (std::size_t kernel = 0; kernel < kernel_names.size(); ++kernel) {
        for (std::size_t device = 0; device < contexts[0].devices.size(); ++device) {
            // create kernel
            kernels[kernel].emplace_back(clCreateKernel(binary_program, kernel_names[kernel].c_str(), &err));
            PLSSVM_OPENCL_ERROR_CHECK(err, fmt::format("error creating OpenCL kernel {} for device {}", kernel_names[kernel], device));
        }
    }

    // release resource
    if (binary_program) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseProgram(binary_program), "error releasing OpenCL binary program resources");
    }
    for (unsigned char *binary : binaries) {
        delete[] binary;
    }

    return kernels;
}

template std::vector<std::vector<kernel>> create_kernel<float>(const std::vector<context> &, const std::vector<std::string> &, const std::vector<std::string> &);
template std::vector<std::vector<kernel>> create_kernel<double>(const std::vector<context> &, const std::vector<std::string> &, const std::vector<std::string> &);

}  // namespace plssvm::opencl::detail