/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"

#include "plssvm/backends/HIP/detail/utility.hip.hpp"  // PLSSVM_HIP_ERROR_CHECK, plssvm::hip::detail::get_device_count
#include "plssvm/backends/HIP/exceptions.hpp"          // plssvm::hip::backend_exception
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT

#include "hip/hip_runtime_api.h"

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <utility>    // std::exchange, std::move, std::swap
#include <vector>     // std::vector

namespace plssvm::hip::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const int device) :
    device_{ device }, size_{ size } {
    if (device_ < 0 || device_ >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), device_) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device_));
    PLSSVM_HIP_ERROR_CHECK(hipMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type)));
}

template <typename T>
device_ptr<T>::device_ptr(device_ptr &&other) noexcept :
    device_{ std::exchange(other.device_, 0) },
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
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device_));
    PLSSVM_HIP_ERROR_CHECK(hipFree(data_));
}

template <typename T>
void device_ptr<T>::swap(device_ptr &other) noexcept {
    std::swap(device_, other.device_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    this->memset(value, pos, size_);
}
template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemset(data_ + pos, value, rcount * sizeof(value_type)));
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

    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), hipMemcpyHostToDevice));
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

    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), hipMemcpyDeviceToHost));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::hip::detail
