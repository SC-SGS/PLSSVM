/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/hipSYCL/detail/device_ptr.hpp"

#include "plssvm/backends/SYCL/detail/constants.hpp"           // forward declaration and namespace alias
#include "plssvm/backends/SYCL/exceptions.hpp"                 // plssvm::sycl::backend_exception
#include "plssvm/backends/SYCL/hipSYCL/detail/queue_impl.hpp"  // plssvm::sycl::detail::queue (PImpl implementation)
#include "plssvm/backends/gpu_device_ptr.hpp"                  // plssvm::detail::gpu_device_ptr
#include "plssvm/detail/assert.hpp"                            // PLSSVM_ASSERT

#include "fmt/core.h"     // fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::malloc_device, sycl::free

#include <algorithm>  // std::min
#include <memory>     // std::unique_ptr

namespace plssvm::hipsycl::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const queue &q) :
    base_type{ size, q } {
    data_ = ::sycl::malloc_device<value_type>(size_, queue_.impl->sycl_queue);
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

    if (pos >= size_) {
        throw sycl::backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rnum_bytes = std::min(num_bytes, (size_ - pos) * sizeof(value_type));
    queue_.impl->sycl_queue.memset(static_cast<void *>(data_ + pos), pattern, rnum_bytes).wait();
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    if (pos >= size_) {
        throw sycl::backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rcount = std::min(count, size_ - pos);
    queue_.impl->sycl_queue.fill(static_cast<void *>(data_ + pos), value, rcount).wait();
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    const size_type rcount = std::min(count, size_ - pos);
    queue_.impl->sycl_queue.copy(data_to_copy, data_ + pos, rcount).wait();
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");
    PLSSVM_ASSERT(queue_.impl != nullptr, "Invalid sycl::queue!");

    const size_type rcount = std::min(count, size_ - pos);
    queue_.impl->sycl_queue.copy(data_ + pos, buffer, rcount).wait();
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::hipsycl::detail