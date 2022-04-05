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

namespace plssvm::opencl::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, command_queue &queue) :
    base_type{ &queue, size } {
    error_code err;
    data_ = clCreateBuffer(queue_->context, CL_MEM_READ_WRITE, size_ * sizeof(value_type), nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err);
}

template <typename T>
device_ptr<T>::~device_ptr() {
    if (data_ != nullptr) {
        PLSSVM_OPENCL_ERROR_CHECK(clReleaseMemObject(data_));
    }
}

template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueFillBuffer(queue_->queue, data_, &value, sizeof(value_type), pos * sizeof(value_type), rcount * sizeof(value_type), 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template <typename T>
void device_ptr<T>::memcpy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueWriteBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), data_to_copy, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template <typename T>
void device_ptr<T>::memcpy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    error_code err;
    err = clEnqueueReadBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), buffer, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err);
    PLSSVM_OPENCL_ERROR_CHECK(clFinish(queue_->queue));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::opencl::detail