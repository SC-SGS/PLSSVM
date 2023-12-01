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
#include "plssvm/backends/gpu_device_ptr.hpp"               // plssvm::detail::gpu_device_ptr
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT

#include "CL/cl.h"     // CL_MEM_READ_WRITE, CL_TRUE, clFinish, clCreateBuffer, clReleaseMemObject, clEnqueueFillBuffer, clEnqueueWriteBuffer, clEnqueueReadBuffer
#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <array>      // std::array

namespace plssvm::opencl::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const command_queue &queue) :
    device_ptr{ { size, 0 }, { 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const std::array<size_type, 2> extents, const command_queue &queue) :
    device_ptr{ extents, { 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const std::array<size_type, 2> extents, std::array<size_type, 2> padding, const command_queue &queue) :
    base_type{ extents, padding, &queue } {
    error_code err{};
    cl_context cont{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue_->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &cont, nullptr));
    data_ = clCreateBuffer(cont, CL_MEM_READ_WRITE, this->size() * sizeof(value_type), nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err);
}

template <typename T>
device_ptr<T>::~device_ptr() {
    if (data_ != nullptr) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseMemObject(data_));
    }
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size()) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, this->size()) };
    }
    const size_type rnum_bytes = std::min(num_bytes, (this->size() - pos) * sizeof(value_type));
    error_code err;
    const auto correct_value = static_cast<unsigned char>(pattern);
    err = clEnqueueFillBuffer(queue_->queue, data_, &correct_value, sizeof(unsigned char), pos * sizeof(value_type), rnum_bytes, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size()) {
        throw backend_exception{ fmt::format("Illegal access in fill!: {} >= {}", pos, this->size()) };
    }

    // run GPU kernel
    const size_type rcount = std::min(count, this->size() - pos);
    error_code err;
    err = clEnqueueFillBuffer(queue_->queue, data_, &value, sizeof(value_type), pos * sizeof(value_type), rcount * sizeof(value_type), 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size() - pos);
    error_code err;
    err = clEnqueueWriteBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), data_to_copy, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size() - pos);
    error_code err;
    err = clEnqueueReadBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), buffer, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::opencl::detail