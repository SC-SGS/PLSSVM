/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"

#include "plssvm/backends/gpu_device_ptr.hpp"                     // plssvm::detail::gpu_device_ptr
#include "plssvm/backends/HIP/detail/utility.hip.hpp"             // PLSSVM_HIP_ERROR_CHECK, plssvm::hip::detail::get_device_count
#include "plssvm/backends/HIP/exceptions.hpp"                     // plssvm::hip::backend_exception
#include "plssvm/backends/HIP/kernel/detail/fill_kernel.hip.hpp"  // plssvm::hip::detail::fill_array
#include "plssvm/detail/assert.hpp"                               // PLSSVM_ASSERT
#include "plssvm/shape.hpp"                                       // plssvm::shape

#include "hip/hip_runtime_api.h"  // HIP runtime functions

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <array>      // std::array
#include <exception>  // std::terminate
#include <iostream>   // std::cerr, std::endl

namespace plssvm::hip::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const queue_type device) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, device } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const queue_type device) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, device } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const plssvm::shape padding, const queue_type device) :
    base_type{ shape, padding, device } {
    if (queue_ < 0 || queue_ >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), queue_) };
    }
    detail::set_device(queue_);
    PLSSVM_HIP_ERROR_CHECK(hipMalloc(&data_, this->size_padded() * sizeof(value_type)))
    this->memset(0);
}

template <typename T>
device_ptr<T>::~device_ptr() {
    // avoid compiler warnings
    try {
        detail::set_device(queue_);
        PLSSVM_HIP_ERROR_CHECK(hipFree(data_))
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

    detail::set_device(queue_);
    const size_type rnum_bytes = std::min(num_bytes, (this->size_padded() - pos) * sizeof(value_type));
    PLSSVM_HIP_ERROR_CHECK(hipMemset(data_ + pos, pattern, rnum_bytes))
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in fill!: {} >= {}", pos, this->size_padded()) };
    }

    detail::set_device(queue_);

    // run GPU kernel
    const size_type rcount = std::min(count, this->size_padded() - pos);
    int block_size = 512;
    int grid_size = (rcount + block_size - 1) / block_size;
    detail::fill_array<<<grid_size, block_size>>>(data_, value, pos, rcount);

    detail::peek_at_last_error();
    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    detail::set_device(queue_);
    const size_type rcount = std::min(count, this->size_padded() - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), hipMemcpyHostToDevice))
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    if (width > spitch) {
        throw backend_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }

    detail::set_device(queue_);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy2D(data_, this->shape_padded().x * sizeof(value_type), data_to_copy, spitch * sizeof(value_type), width * sizeof(value_type), height, hipMemcpyHostToDevice));
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    detail::set_device(queue_);
    const size_type rcount = std::min(count, this->size_padded() - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), hipMemcpyDeviceToHost))
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    detail::set_device(queue_);
    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(target.get(), data_ + pos, rcount * sizeof(value_type), hipMemcpyDeviceToDevice))
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::hip::detail
