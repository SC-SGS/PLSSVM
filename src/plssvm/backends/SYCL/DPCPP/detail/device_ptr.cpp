/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/DPCPP/detail/device_ptr.hpp"

#include "plssvm/backends/gpu_device_ptr.hpp"                // plssvm::detail::gpu_device_ptr
#include "plssvm/backends/SYCL/DPCPP/detail/queue_impl.hpp"  // plssvm::dpcpp::detail::queue (PImpl implementation)
#include "plssvm/backends/SYCL/exceptions.hpp"               // plssvm::dpcpp::backend_exception
#include "plssvm/detail/assert.hpp"                          // PLSSVM_ASSERT

#include "sycl/sycl.hpp"  // ::sycl::malloc_device, ::sycl::free

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <vector>     // std::vector

namespace plssvm::dpcpp::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const queue &q) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, q } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const queue &q) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, q } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, plssvm::shape padding, const queue &q) :
    base_type{ shape, padding, q } {
    data_ = ::sycl::malloc_device<value_type>(this->size_padded(), queue_.impl->sycl_queue);
    this->memset(0);
}

template <typename T>
device_ptr<T>::~device_ptr() {
    if (queue_.impl != nullptr) {
        ::sycl::free(static_cast<void *>(data_), queue_.impl->sycl_queue);
    }
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, this->size_padded()) };
    }
    const size_type rnum_bytes = std::min(num_bytes, (this->size_padded() - pos) * sizeof(value_type));
    queue_.impl->sycl_queue.memset(static_cast<void *>(data_ + pos), pattern, rnum_bytes).wait();
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in fill!: {} >= {}", pos, this->size_padded()) };
    }
    const size_type rcount = std::min(count, this->size_padded() - pos);
    queue_.impl->sycl_queue.fill(static_cast<void *>(data_ + pos), value, rcount).wait();
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    queue_.impl->sycl_queue.copy(data_to_copy, data_ + pos, rcount).wait();
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    if (width > spitch) {
        throw backend_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }

    // if available, use the DPC++ ext_oneapi_copy2d extension, otherwise fallback to a temporary
#if defined(SYCL_EXT_ONEAPI_MEMCPY2D)
    queue_.impl->sycl_queue.ext_oneapi_copy2d(data_to_copy, spitch, data_, this->shape_padded().x, width, height).wait();
#else
    if (spitch == width) {
        // can use normal copy since we have no line strides
        this->copy_to_device(data_to_copy, 0, width * height);
    } else {
        std::vector<value_type> temp(this->shape_padded().x * height, value_type{ 0.0 });
        value_type *pos = temp.data();
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(pos, data_to_copy + row * spitch, width * sizeof(value_type));
            pos += this->shape_padded().x;
        }
        this->copy_to_device(temp);
    }
#endif
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    queue_.impl->sycl_queue.copy(data_ + pos, buffer, rcount).wait();
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }

    // TODO: direct copy between devices in DPC++ currently not possible
    std::vector<value_type> temp(rcount);
    this->copy_to_host(temp, pos, rcount);
    target.copy_to_device(temp);
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::dpcpp::detail
