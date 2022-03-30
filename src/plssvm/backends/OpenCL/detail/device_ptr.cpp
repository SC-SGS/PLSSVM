/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"

#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/error_code.hpp"     // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // PLSSVM_OPENCL_ERROR_CHECK
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT

#include "CL/cl.h"     // CL_MEM_READ_WRITE, CL_TRUE, clFinish, clCreateBuffer, clReleaseMemObject, clEnqueueFillBuffer, clEnqueueWriteBuffer, clEnqueueReadBuffer
#include "fmt/core.h"  // fmt::format

#include <algorithm>    // std::min
#include <map>          // std::map
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::exchange, std::move, std::swap
#include <vector>       // std::vector

namespace plssvm::opencl::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, cl_command_queue &queue) :
    queue_{ queue }, size_{ size } {
    error_code err;
    cl_context cont;
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &cont, nullptr));
    data_ = clCreateBuffer(cont, CL_MEM_READ_WRITE, size_ * sizeof(value_type), nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err);
}

template <typename T>
device_ptr<T>::device_ptr(device_ptr &&other) noexcept :
    queue_{ other.queue_ },
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
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (buffer.size() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->memcpy_to_host(buffer.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer, const size_type pos, const size_type count) const {
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