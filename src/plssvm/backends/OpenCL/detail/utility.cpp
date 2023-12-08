/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/utility.hpp"

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/context.hpp"        // plssvm::opencl::detail::context
#include "plssvm/backends/OpenCL/detail/error_code.hpp"     // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/kernel.hpp"         // plssvm::opencl::detail::compute_kernel_name, plssvm::opencl::detail::kernel
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                             // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/logging.hpp"                        // plssvm::detail::log
#include "plssvm/detail/sha256.hpp"                         // plssvm::detail::sha256
#include "plssvm/detail/string_conversion.hpp"              // plssvm::detail::extract_first_integer_from_string
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all, plssvm::detail::to_lower_case, plssvm::detail::contains
#include "plssvm/detail/utility.hpp"                        // plssvm::detail::erase_if
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::unsupported_kernel_type_exception, plssvm::invalid_file_format_exception
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                      // plssvm::verbosity_level

#include "CL/cl.h"        // cl_program, cl_platform_id, cl_device_id, cl_uint, cl_device_type, cl_context,
                          // CL_DEVICE_NAME, CL_QUEUE_DEVICE, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_VENDOR, CL_PROGRAM_BUILD_LOG, CL_PROGRAM_BINARY_SIZES, CL_PROGRAM_BINARIES,
                          // clCreateProgramWithSource, clBuildProgram, clGetProgramBuildInfo, clGetProgramInfo, clCreateKernel, clReleaseProgram, clCreateProgramWithBinary,
                          //  clSetKernelArg, clEnqueueNDRangeKernel, clFinish, clGetPlatformIDs, clGetDeviceIDs, clGetDeviceInfo, clCreateContext
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "fmt/std.h"      // format std::filesystem::path

#include <algorithm>    // std::count_if
#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::{path, temp_directory_path, exists, directory_iterator, directory_entry}
#include <fstream>      // std::ifstream, std::ofstream
#include <functional>   // std::hash
#include <ios>          // std::ios_base, std::streamsize
#include <iostream>     // std::cout, std::endl
#include <iterator>     // std::istreambuf_iterator
#include <limits>       // std::numeric_limits
#include <map>          // std::map
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tie
#include <utility>      // std::pair, std::make_pair, std::move
#include <vector>       // std::vector

namespace plssvm::opencl::detail {

void device_assert(const error_code ec, const std::string_view msg) {
    if (!ec) {
        if (msg.empty()) {
            throw backend_exception{ fmt::format("OpenCL assert '{}' ({})!", ec.message(), ec.value()) };
        } else {
            throw backend_exception{ fmt::format("OpenCL assert '{}' ({}): {}!", ec.message(), ec.value(), msg) };
        }
    }
}

[[nodiscard]] std::pair<std::vector<context>, target_platform> get_contexts(target_platform target) {
    error_code err;

    // function to add a key value pair to a map, where the value is added to a std::vector
    const auto add_to_map = [](auto &map, const auto &key, auto value) {
        // if key currently doesn't exist, add a new std::vector
        if (map.count(key) == 0) {
            map[key] = std::vector<decltype(value)>();
        }
        // add value to std::vector represented by key
        map[key].push_back(value);
    };

    // iterate over all platforms and save all available devices
    std::map<std::pair<cl_platform_id, target_platform>, std::vector<cl_device_id>> platform_devices;
    // get number of platforms
    cl_uint num_platforms{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms), "error retrieving the number of available platforms");
    // get platforms
    std::vector<cl_platform_id> platform_ids(num_platforms);
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr), "error retrieving platform IDs");

    // get the available target_platforms
    const std::vector<target_platform> available_target_platforms = list_available_target_platforms();

    // enumerate all available platforms and retrieve the associated devices
    for (const cl_platform_id &platform : platform_ids) {
        // get devices associated with current platform
        cl_uint num_devices{};
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices), "error retrieving the number of devices");
        // get devices
        std::vector<cl_device_id> device_ids(num_devices);
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr), "error retrieving device IDs");

        for (const cl_device_id &device : device_ids) {
            // get device type
            cl_device_type device_type{};
            PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr), "error retrieving the device type");

            if (device_type == CL_DEVICE_TYPE_CPU) {
                // the current device is a CPU
                // -> check if the CPU target has been enabled
                if (::plssvm::detail::contains(available_target_platforms, target_platform::cpu)) {
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::cpu), device);
                }
            } else if (device_type == CL_DEVICE_TYPE_GPU) {
                // the current device is a GPU
                // get vendor string
                std::size_t vendor_string_size{};
                PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, 0, nullptr, &vendor_string_size), "error retrieving device vendor name size");
                std::string vendor_string(vendor_string_size, '\0');
                PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendor_string_size, vendor_string.data(), nullptr), "error retrieving device vendor name");
                // convert vendor name to lower case
                ::plssvm::detail::to_lower_case(vendor_string);

                // check vendor string and insert to correct target platform
                if (::plssvm::detail::contains(vendor_string, "nvidia") && ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_nvidia)) {
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_nvidia), device);
                } else if ((::plssvm::detail::contains(vendor_string, "amd") || ::plssvm::detail::contains(vendor_string, "advanced micro devices"))
                           && ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_amd)) {
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_amd), device);
                } else if (::plssvm::detail::contains(vendor_string, "intel") && ::plssvm::detail::contains(available_target_platforms, target_platform::gpu_intel)) {
                    add_to_map(platform_devices, std::make_pair(platform, target_platform::gpu_intel), device);
                }
            }
        }
    }

    // determine target if provided target_platform is automatic
    if (target == target_platform::automatic) {
        // get the target_platforms available on this system from the platform_devices map
        std::vector<target_platform> system_devices;
        system_devices.reserve(platform_devices.size());
        for (const auto &[key, value] : platform_devices) {
            system_devices.push_back(key.second);
        }
        // determine the target_platform
        target = determine_default_target_platform(system_devices);
    }

    // only interested in devices on the target platform
    ::plssvm::detail::erase_if(platform_devices, [target](const auto &item) {
        // target_platform of the current OpenCL platform must match the requested target
        return item.first.second != target;
    });

    // create one context for each leftover target platform
    std::vector<context> contexts;
    for (auto &[platform, devices] : platform_devices) {
        // create context and associated OpenCL platform with it
        std::array<cl_context_properties, 3> context_properties = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform.first, 0 };
        cl_context cont = clCreateContext(context_properties.data(), static_cast<cl_uint>(devices.size()), devices.data(), nullptr, nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the OpenCL context");
        // add OpenCL context to vector of context wrappers
        contexts.emplace_back(cont, platform.first, std::move(devices));
    }

    return std::make_pair(std::move(contexts), target);
}

void device_synchronize(const command_queue &queue) {
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue));
}

std::string get_device_name(const command_queue &queue) {
    // get device
    cl_device_id device_id{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(cl_device_id), &device_id, nullptr), "error obtaining device");
    // get device name
    std::size_t name_length{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &name_length), "error obtaining device name size");
    std::string device_name(name_length - 1, '\0');
    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_NAME, name_length, device_name.data(), nullptr), "error obtaining device name");
    return device_name;
}

std::vector<std::pair<compute_kernel_name, std::string>> kernel_type_to_function_names(const kernel_function_type kernel) {
    // since the correct predict kernel function cannot be determined during construction, add all predict kernels
    std::vector<std::pair<compute_kernel_name, std::string>> kernels{
        std::make_pair(compute_kernel_name::assemble_kernel_matrix_explicit, fmt::format("device_kernel_assembly_{}", kernel)),
        std::make_pair(compute_kernel_name::predict_kernel_linear, "device_kernel_predict_linear"),
        std::make_pair(compute_kernel_name::predict_kernel_polynomial, "device_kernel_predict_polynomial"),
        std::make_pair(compute_kernel_name::predict_kernel_rbf, "device_kernel_predict_rbf"),
        std::make_pair(compute_kernel_name::w_kernel, "device_kernel_w_linear")
    };

#if defined(PLSSVM_USE_GEMM)
    kernels.emplace_back(compute_kernel_name::gemm_kernel_explicit, "device_kernel_gemm");
#else
    kernels.emplace_back(compute_kernel_name::symm_kernel_explicit, "device_kernel_symm");
#endif

    return kernels;
}

std::vector<command_queue> create_command_queues(const std::vector<context> &contexts, const target_platform target, const std::vector<std::pair<compute_kernel_name, std::string>> &kernel_names) {
    std::vector<command_queue> queues;
    for (std::vector<cl_device_id>::size_type device = 0; device < contexts[0].devices.size(); ++device) {
        queues.emplace_back(contexts[0], contexts[0].devices[device]);
    }
    PLSSVM_ASSERT(!queues.empty(), "At least one command queue must be available!");

    const auto cl_build_program_error_message = [](cl_program prog, cl_device_id device, const std::size_t device_idx) {
        // determine the size of the log
        std::size_t log_size{};
        PLSSVM_OPENCL_ERROR_CHECK(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size), "error retrieving the program build log size");
        if (log_size > 0) {
            // allocate memory for the log
            std::string log(log_size, ' ');
            // get the log
            PLSSVM_OPENCL_ERROR_CHECK(clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr), "error retrieving the program build log");
            // print the log
            std::cerr << fmt::format("error building OpenCL program on device {}:\n{}", device_idx, log) << std::endl;
        }
    };

    error_code err, err_bin;

    // note: unsigned long long may NOT be used in an OpenCL kernel (use ulong instead)
    // note: real_type temp{ 0.0 } may NOT be used in an OpenCL kernel (use real_type temp = 0.0 instead)

    // create one string from each OpenCL source file in the OpenCL include directory
    const std::filesystem::path base_path{ PLSSVM_OPENCL_KERNEL_SOURCE_DIR };
    std::string kernel_src_string{};
    // note: the detail/atomics.cl file must be included first!
    for (const auto &path : { base_path / "detail/atomics.cl", base_path / "cg_explicit/blas.cl", base_path / "cg_explicit/kernel_matrix_assembly.cl", base_path / "predict_kernel.cl" }) {
        std::ifstream file{ base_path / path };
        kernel_src_string.append((std::istreambuf_iterator<char>{ file }),
                                 std::istreambuf_iterator<char>{});
    }

    // replace types in kernel_src_string
    ::plssvm::detail::replace_all(kernel_src_string, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
    // replace constants in kernel_src_string
    ::plssvm::detail::replace_all(kernel_src_string, "THREAD_BLOCK_SIZE", fmt::format("{}", THREAD_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "FEATURE_BLOCK_SIZE", fmt::format("{}", FEATURE_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "INTERNAL_BLOCK_SIZE", fmt::format("{}", INTERNAL_BLOCK_SIZE));
    ::plssvm::detail::replace_all(kernel_src_string, "PADDING_SIZE", fmt::format("{}", PADDING_SIZE));

    // append number of device to influence checksum calculation
    kernel_src_string.append(fmt::format("\n// num_devices: {}\n// OpenCL library: {}\n// GEMM: {}", contexts[0].devices.size(), PLSSVM_OPENCL_LIBRARY, PLSSVM_IS_DEFINED(PLSSVM_USE_GEMM)));

    // create source code hash
    const std::string checksum = plssvm::detail::sha256{}(kernel_src_string);

    // convert string to const char*
    const char *kernel_src_ptr = kernel_src_string.c_str();

    // data to build the final OpenCL program
    std::vector<std::size_t> binary_sizes(contexts[0].devices.size());
    std::vector<unsigned char *> binaries(contexts[0].devices.size());

    // create caching folder in the temporary directory and change the permissions such that everybody has read/write access
    const std::filesystem::path cache_dir_name = std::filesystem::temp_directory_path() / "plssvm_opencl_cache" / fmt::format("{}_{}_{}_{}", checksum, target, PLSSVM_CMAKE_BUILD_TYPE, ::plssvm::detail::arithmetic_type_name<real_type>());

    // potential reasons why OpenCL caching could fail
    enum class caching_status {
        success,
        error_no_cached_files,
        error_invalid_number_of_cached_files,
    };
    // message associated with the failed caching reason
    const auto caching_status_to_string = [](const caching_status status) {
        switch (status) {
            case caching_status::error_no_cached_files:
                return "no cached files exist (checksum missmatch)";
            case caching_status::error_invalid_number_of_cached_files:
                return "invalid number of cached files";
            default:
                return "";
        }
    };

    // assume caching was successful
    caching_status use_cached_binaries = caching_status::success;

    // check if cache directory exists
    if (!std::filesystem::exists(cache_dir_name)) {
        use_cached_binaries = caching_status::error_no_cached_files;
    }
    // if the cache directory exists, check the number of files
    if (use_cached_binaries == caching_status::success) {
        // get directory iterator
        auto dirIter = std::filesystem::directory_iterator(cache_dir_name);
        // get files in directory
        if (static_cast<std::size_t>(std::count_if(begin(dirIter), end(dirIter), [](const auto &entry) { return entry.is_regular_file(); })) != contexts[0].devices.size()) {
            use_cached_binaries = caching_status::error_invalid_number_of_cached_files;
        }
    }

    if (use_cached_binaries != caching_status::success) {
        plssvm::detail::log(verbosity_level::full,
                            "Building OpenCL kernels from source (reason: {}).\n",
                            caching_status_to_string(use_cached_binaries));

        // create and build program
        cl_program program = clCreateProgramWithSource(contexts[0], 1, &kernel_src_ptr, nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error creating program from source");
#if defined(PLSSVM_USE_GEMM)
        err = clBuildProgram(program, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros -DPLSSVM_USE_GEMM", nullptr, nullptr);
#else
        err = clBuildProgram(program, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), "-cl-fast-relaxed-math -cl-mad-enable -cl-no-signed-zeros", nullptr, nullptr);
#endif
        if (!err) {
            // check all devices for errors
            for (std::vector<context>::size_type device = 0; device < contexts[0].devices.size(); ++device) {
                cl_build_program_error_message(program, contexts[0].devices[device], device);
            }
        }

        // get sizes of binaries
        err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, contexts[0].devices.size() * sizeof(std::size_t), binary_sizes.data(), nullptr);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error retrieving the kernel (binary) kernel sizes");
        for (std::vector<unsigned char *>::size_type i = 0; i < binaries.size(); ++i) {
            binaries[i] = new unsigned char[binary_sizes[i]];
        }

        // get binaries
        err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, contexts[0].devices.size() * sizeof(unsigned char *), binaries.data(), nullptr);
        PLSSVM_OPENCL_ERROR_CHECK(err, "error retrieving the kernel binaries");

        // write binaries to file
        if (!std::filesystem::exists(cache_dir_name)) {
            std::filesystem::create_directories(cache_dir_name);
        }
        [[maybe_unused]] std::error_code file_perm_error;
        const auto super_cache_dir_name = std::filesystem::temp_directory_path() / "plssvm_opencl_cache";
        std::filesystem::permissions(super_cache_dir_name, std::filesystem::perms::all, std::filesystem::perm_options::add, file_perm_error);
        for (const std::filesystem::path &path : std::filesystem::recursive_directory_iterator{ super_cache_dir_name }) {
            std::filesystem::permissions(path, std::filesystem::perms::all, file_perm_error);
        }

        for (std::vector<std::size_t>::size_type i = 0; i < binary_sizes.size(); ++i) {
            std::ofstream out{ cache_dir_name / fmt::format("device_{}.bin", i) };
            PLSSVM_ASSERT(out.good(), fmt::format("couldn't create binary cache file ({}) for device {}", cache_dir_name / fmt::format("device_{}.bin", i), i));
            out.write(reinterpret_cast<char *>(binaries[i]), binary_sizes[i]);
        }
        plssvm::detail::log(verbosity_level::full,
                            "Cached OpenCL kernel binaries in {}.\n",
                            cache_dir_name);

        // release resource
        if (program) {
            PLSSVM_OPENCL_ERROR_CHECK(clReleaseProgram(program), "error releasing OpenCL program resources");
        }
    } else {
        plssvm::detail::log(verbosity_level::full,
                            "Using cached OpenCL kernel binaries from {}.\n",
                            cache_dir_name);

        const auto common_read_file = [](const std::filesystem::path &file) -> std::pair<unsigned char *, std::size_t> {
            std::ifstream f{ file };

            // touch all characters in file
            f.ignore(std::numeric_limits<std::streamsize>::max());
            // get number of visited characters
            const std::streamsize num_bytes = f.gcount();
            // since ignore will have set eof
            f.clear();
            // jump to file start
            f.seekg(0, std::ios_base::beg);

            // allocate the necessary buffer
            char *file_content = new char[num_bytes];
            // read the whole file in one go
            f.read(file_content, num_bytes);
            return std::make_pair(reinterpret_cast<unsigned char *>(file_content), static_cast<std::size_t>(num_bytes));
        };

        // iterate over directory and read kernels into binary file
        auto dirIter = std::filesystem::directory_iterator(cache_dir_name);
        for (const std::filesystem::directory_entry &entry : dirIter) {
            if (entry.is_regular_file()) {
                const auto i = ::plssvm::detail::extract_first_integer_from_string<std::size_t>(entry.path().filename().string());
                std::tie(binaries[i], binary_sizes[i]) = common_read_file(entry.path());
            }
        }
    }

    // build from binaries
    cl_program binary_program = clCreateProgramWithBinary(contexts[0], static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), binary_sizes.data(), const_cast<const unsigned char **>(binaries.data()), &err_bin, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err_bin, "error loading binaries");
    PLSSVM_OPENCL_ERROR_CHECK(err, "error creating binary program");
    err = clBuildProgram(binary_program, static_cast<cl_uint>(contexts[0].devices.size()), contexts[0].devices.data(), nullptr, nullptr, nullptr);
    if (!err) {
        // check all devices for errors
        for (std::vector<context>::size_type device = 0; device < contexts[0].devices.size(); ++device) {
            cl_build_program_error_message(binary_program, contexts[0].devices[device], device);
        }
    }

    // build all kernels, one for each device
    for (std::vector<cl_device_id>::size_type device = 0; device < contexts[0].devices.size(); ++device) {
        for (std::vector<std::vector<kernel>>::size_type i = 0; i < kernel_names.size(); ++i) {
            // create kernel
            queues[device].add_kernel(kernel_names[i].first, kernel{ clCreateKernel(binary_program, kernel_names[i].second.c_str(), &err) });
            PLSSVM_OPENCL_ERROR_CHECK(err, fmt::format("error creating OpenCL kernel {} for device {}", kernel_names[i].second, device));
        }
    }

    // release resource
    if (binary_program) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseProgram(binary_program), "error releasing OpenCL binary program resources");
    }
    for (unsigned char *binary : binaries) {
        delete[] binary;
    }

    return queues;
}

}  // namespace plssvm::opencl::detail