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
#include "plssvm/backends/gpu_device_ptr.hpp"          // plssvm::detail::gpu_device_ptr
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT

#include "hip/hip_runtime_api.h"

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min

namespace plssvm::hip::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const int device) :
    base_type{ device, size } {
    if (queue_ < 0 || queue_ >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), queue_) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(queue_));
    PLSSVM_HIP_ERROR_CHECK(hipMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type)));
}

template <typename T>
device_ptr<T>::~device_ptr() {
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(queue_));
    PLSSVM_HIP_ERROR_CHECK(hipFree(data_));
}

template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(queue_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemset(data_ + pos, value, rcount * sizeof(value_type)));
}

template <typename T>
void device_ptr<T>::memcpy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(queue_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), hipMemcpyHostToDevice));
}

template <typename T>
void device_ptr<T>::memcpy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(queue_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_HIP_ERROR_CHECK(hipMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), hipMemcpyDeviceToHost));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::hip::detail
