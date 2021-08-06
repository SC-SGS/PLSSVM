/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"

#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/exceptions.hpp"         // plssvm::opencl::backend_exception
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::to_lower_case, plssvm::detail::contains
#include "plssvm/target_platform.hpp"                    // plssvm::target_platform

#include "CL/cl.h"     // cl_command_queue, cl_context, clGetCommandQueueInfo, clCreateBuffer, clReleaseMemObject, clEnqueueFillBuffer
                       // clFinish, clEnqueueWriteBuffer, clEnqueueReadBuffer
#include "fmt/core.h"  // fmt::format

#include <algorithm>    // std::min
#include <string_view>  // std::string_view
#include <utility>      // std::exchange, std::move, std::swap
#include <vector>       // std::vector

#define PLSSVM_OPENCL_ERROR_CHECK(err) plssvm::opencl::detail::device_assert((err))

namespace plssvm::opencl::detail {

inline void device_assert(const error_code ec) {
    if (!ec) {
        throw backend_exception{ fmt::format("OpenCL assert ({})", ec) };
    }
}

[[nodiscard]] std::vector<cl_command_queue> get_device_list_impl(const target_platform target) {
    std::vector<cl_device_id> target_devices;

    // get number of platforms
    cl_uint num_platforms;
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));
    // get platforms
    std::vector<cl_platform_id> platform_ids(num_platforms);
    PLSSVM_OPENCL_ERROR_CHECK(clGetPlatformIDs(num_platforms, platform_ids.data(), nullptr));

    for (const cl_platform_id &platform : platform_ids) {
        // get number of devices
        cl_uint num_devices;
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
        // get devices
        std::vector<cl_device_id> device_ids(num_devices);
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids.data(), nullptr));

        for (const cl_device_id &device : device_ids) {
            cl_device_type device_type;
            PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr));
            if (target == target_platform::cpu && device_type == CL_DEVICE_TYPE_CPU) {
                // select CPU device
                target_devices.emplace_back(device);
            } else {
                // must be a GPU device
                if (device_type == CL_DEVICE_TYPE_GPU) {
                    // get vendor name of current GPU
                    std::string vendor_string(128, '\0');
                    PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(device, CL_DEVICE_VENDOR, vendor_string.size() * sizeof(char), vendor_string.data(), nullptr));
                    vendor_string = vendor_string.substr(0, vendor_string.find_first_of('\0'));
                    // convert vendor name to lower case
                    ::plssvm::detail::to_lower_case(vendor_string);

                    switch (target) {
                        case target_platform::gpu_nvidia:
                            if (::plssvm::detail::contains(vendor_string, "nvidia")) {
                                target_devices.emplace_back(device);
                            }
                            break;
                        case target_platform::gpu_amd:
                            if (::plssvm::detail::contains(vendor_string, "amd")) {
                                target_devices.emplace_back(device);
                            }
                            break;
                        case target_platform::gpu_intel:
                            if (::plssvm::detail::contains(vendor_string, "intel")) {
                                target_devices.emplace_back(device);
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
    // TODO: context
    cl_context context;
    if (!target_devices.empty()) {
        cl_platform_id platform;
        PLSSVM_OPENCL_ERROR_CHECK(clGetDeviceInfo(target_devices[0], CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, nullptr));
        error_code err;
        context = clCreateContext(nullptr, static_cast<cl_uint>(target_devices.size()), target_devices.data(), nullptr, nullptr, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err);
    }
    // convert device ids to command queues
    std::vector<cl_command_queue> command_queues(target_devices.size());
    error_code err;
    for (std::size_t device = 0; device < target_devices.size(); ++device) {
        command_queues[device] = clCreateCommandQueue(context, target_devices[device], 0, &err);
        PLSSVM_OPENCL_ERROR_CHECK(err);
    }

    return command_queues;
}

std::vector<cl_command_queue> get_device_list(const target_platform target) {
    if (target != target_platform::automatic) {
        return get_device_list_impl(target);
    } else {
        std::vector<cl_command_queue> target_devices = get_device_list_impl(target_platform::gpu_nvidia);
        if (target_devices.empty()) {
            target_devices = get_device_list_impl(target_platform::gpu_amd);
            if (target_devices.empty()) {
                target_devices = get_device_list_impl(target_platform::gpu_intel);
                if (target_devices.empty()) {
                    target_devices = get_device_list_impl(target_platform::cpu);
                }
            }
        }
        return target_devices;
    }
}

void device_synchronize(cl_command_queue queue) {
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue));
}

template <typename T>
device_ptr<T>::device_ptr(const size_type size, cl_command_queue queue) :
    queue_{ queue }, size_{ size } {
    cl_context context;  // TODO: clReleaseContext?
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue_, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, nullptr));

    error_code err;
    data_ = clCreateBuffer(context, CL_MEM_READ_WRITE, size_ * sizeof(value_type), nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err);
}

template <typename T>
device_ptr<T>::device_ptr(device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, nullptr) },
    data_{ std::exchange(other.data_, nullptr) },
    size_{ std::exchange(other.size_, 0) } {}

template <typename T>
device_ptr<T> &device_ptr<T>::operator=(device_ptr &&other) noexcept {
    device_ptr tmp{ std::move(other) };
    this->swap(tmp);
    return *this;
}

template <typename T>
device_ptr<T>::~device_ptr() {
    if (data_ != nullptr) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseMemObject(data_));
    }
}

template <typename T>
void device_ptr<T>::swap(device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

template <typename T>
void device_ptr<T>::memset(const value_type value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memset(value, pos, size_);
}
template <typename T>
void device_ptr<T>::memset(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueFillBuffer(queue_, data_, &value, sizeof(value_type), pos * sizeof(value_type), rcount * sizeof(value_type), 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_));
}

template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (data_to_copy.size() < rcount) {
        throw backend_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->memcpy_to_device(data_to_copy.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueWriteBuffer(queue_, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), data_to_copy, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_));
}

template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (buffer.size() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->memcpy_to_host(buffer.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueReadBuffer(queue_, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), buffer, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::opencl::detail