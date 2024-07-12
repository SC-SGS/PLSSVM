/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"

#include "plssvm/backends/gpu_device_ptr.hpp"               // plssvm::detail::gpu_device_ptr
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/backends/OpenCL/detail/error_code.hpp"     // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/utility.hpp"        // PLSSVM_OPENCL_ERROR_CHECK
#include "plssvm/backends/OpenCL/exceptions.hpp"            // plssvm::opencl::backend_exception
#include "plssvm/detail/assert.hpp"                         // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/shape.hpp"                                 // plssvm::shape

#include "CL/cl.h"  // CL_MEM_READ_WRITE, CL_TRUE, clFinish, clCreateBuffer, clReleaseMemObject, clEnqueueFillBuffer, clEnqueueWriteBuffer, clEnqueueReadBuffer

#include "fmt/format.h"  // fmt::format

#include <algorithm>  // std::min
#include <array>      // std::array
#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cerr, std::endl
#include <vector>     // std::vector

namespace plssvm::opencl::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const command_queue &queue) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const command_queue &queue) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const plssvm::shape padding, const command_queue &queue) :
    base_type{ shape, padding, &queue } {
    error_code err{};
    cl_context cont{};
    PLSSVM_OPENCL_ERROR_CHECK(clGetCommandQueueInfo(queue_->queue, CL_QUEUE_CONTEXT, sizeof(cl_context), static_cast<void *>(&cont), nullptr), "error retrieving the command queue context")
    data_ = clCreateBuffer(cont, CL_MEM_READ_WRITE, this->size_padded() * sizeof(value_type), nullptr, &err);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error creating the buffer")
    this->memset(0);
}

template <typename T>
device_ptr<T>::~device_ptr() {
    // avoid compiler warnings
    try {
        if (data_ != nullptr) {
            PLSSVM_OPENCL_ERROR_CHECK(clReleaseMemObject(data_), "error releasing the buffer")
        }
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, this->size_padded()) };
    }
    const size_type rnum_bytes = std::min(num_bytes, (this->size_padded() - pos) * sizeof(value_type));
    error_code err;
    const auto correct_value = static_cast<unsigned char>(pattern);
    err = clEnqueueFillBuffer(queue_->queue, data_, &correct_value, sizeof(unsigned char), pos * sizeof(value_type), rnum_bytes, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error filling the buffer via memset")
    device_synchronize(*queue_);
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in fill!: {} >= {}", pos, this->size_padded()) };
    }

    // run GPU kernel
    const size_type rcount = std::min(count, this->size_padded() - pos);
    error_code err;
    err = clEnqueueFillBuffer(queue_->queue, data_, &value, sizeof(value_type), pos * sizeof(value_type), rcount * sizeof(value_type), 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error filling the buffer via fill")
    device_synchronize(*queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    error_code err;
    err = clEnqueueWriteBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), data_to_copy, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error copying the data to the device buffer")
    device_synchronize(*queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    if (width > spitch) {
        throw backend_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }

    const std::array<std::size_t, 3> buffer_origin{ 0, 0, 0 };
    const std::array<std::size_t, 3> host_origin{ 0, 0, 0 };
    const std::array<std::size_t, 3> region{ width * sizeof(value_type), height, 1 };
    const std::size_t buffer_row_pitch = this->shape_padded().x * sizeof(value_type);
    const std::size_t buffer_slice_pitch = 0;
    const std::size_t host_row_pitch = spitch * sizeof(value_type);
    const std::size_t host_slice_pitch = 0;

    error_code err;
    err = clEnqueueWriteBufferRect(queue_->queue, data_, CL_TRUE, buffer_origin.data(), host_origin.data(), region.data(), buffer_row_pitch, buffer_slice_pitch, host_row_pitch, host_slice_pitch, data_to_copy, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error copying the strided data to the device buffer")
    device_synchronize(*queue_);
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    error_code err;
    err = clEnqueueReadBuffer(queue_->queue, data_, CL_TRUE, pos * sizeof(value_type), rcount * sizeof(value_type), buffer, 0, nullptr, nullptr);
    PLSSVM_OPENCL_ERROR_CHECK(err, "error copying the data from the device buffer")
    device_synchronize(*queue_);
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }

    // TODO: direct copy between devices in OpenCL currently not possible
    std::vector<value_type> temp(rcount);
    this->copy_to_host(temp, pos, rcount);
    target.copy_to_device(temp);
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::opencl::detail
