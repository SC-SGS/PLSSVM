/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/device_ptr.hpp"

#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/detail/constants.hpp"  // forward declaration and namespace alias
#include "plssvm/backends/@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@/exceptions.hpp"        // plssvm::@PLSSVM_SYCL_BACKEND_INCLUDE_NAME@::backend_exception
#include "plssvm/backends/gpu_device_ptr.hpp"                                       // plssvm::detail::gpu_device_ptr
#include "plssvm/detail/assert.hpp"                                                 // PLSSVM_ASSERT

#include "fmt/core.h"     // fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::malloc_device, sycl::free

#include <algorithm>  // std::min
#include <memory>     // std::unique_ptr

namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail {

    template <typename T>
    device_ptr<T>::device_ptr(const size_type size, std::unique_ptr<detail::sycl::queue> &queue) :
        base_type{ queue.get(), size } {
        data_ = detail::sycl::malloc_device<value_type>(size_, *queue_);
    }

    template <typename T>
    device_ptr<T>::~device_ptr() {
        if (queue_ != nullptr) {
            detail::sycl::free(static_cast<void *>(data_), *queue_);
        }
    }

    template <typename T>
    void device_ptr<T>::memset(const int value, const size_type pos, const size_type count) {
        PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
        PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

        if (pos >= size_) {
            throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
        }
        const size_type rcount = std::min(count, size_ - pos);
        queue_->memset(static_cast<void *>(data_ + pos), value, rcount * sizeof(value_type)).wait();
    }

    template <typename T>
    void device_ptr<T>::memcpy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
        PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
        PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

        const size_type rcount = std::min(count, size_ - pos);
        queue_->copy(data_to_copy, data_, rcount).wait();
    }

    template <typename T>
    void device_ptr<T>::memcpy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
        PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
        PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

        const size_type rcount = std::min(count, size_ - pos);
        queue_->copy(data_, buffer, rcount).wait();
    }

    template class device_ptr<float>;
    template class device_ptr<double>;

}  // namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail