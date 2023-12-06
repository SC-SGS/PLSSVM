/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/gpu_device_ptr.hpp"

// no includes for CUDA and HIP since they both use simple ints

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    // used for explicitly instantiating the OpenCL backend
    #include "CL/cl.h"                                          // cl_mem
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // used for explicitly instantiating the SYCL backend
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
        #include "plssvm/backends/SYCL/DPCPP/detail/queue.hpp"  // plssvm::dpcpp::detail::queue
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
        #include "plssvm/backends/SYCL/AdaptiveCpp/detail/queue.hpp"  // plssvm::adaptivecpp::detail::queue
    #endif
#endif

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::gpu_device_ptr_exception

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <array>      // std::array
#include <memory>     // std::addressof
#include <utility>    // std::exchange, std::swap
#include <vector>     // std::vector

namespace plssvm::detail {

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(size_type size, const queue_type queue) :
    queue_{ queue }, extents_{ { size, 0 } } {}

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(std::array<size_type, 2> extents, const queue_type queue) :
    queue_{ queue }, extents_{ extents } {}

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(std::array<size_type, 2> extents, std::array<size_type, 2> padding, const queue_type queue) :
    queue_{ queue }, extents_{ extents }, padding_{ padding } {}

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(gpu_device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, queue_type{}) },
    data_{ std::exchange(other.data_, device_pointer_type{}) },
    extents_{ std::exchange(other.extents_, std::array<size_type, 2>{}) },
    padding_{ std::exchange(other.padding_, std::array<size_type, 2>{}) } {}

template <typename T, typename queue_t, typename device_pointer_t>
auto gpu_device_ptr<T, queue_t, device_pointer_t>::operator=(gpu_device_ptr &&other) noexcept -> gpu_device_ptr & {
    // guard against self-assignment
    if (this != std::addressof(other)) {
        queue_ = std::exchange(other.queue_, queue_type{});
        data_ = std::exchange(other.data_, device_pointer_type{});
        extents_ = std::exchange(other.extents_, std::array<size_type, 2>{});
        padding_ = std::exchange(other.padding_, std::array<size_type, 2>{});
    }
    return *this;
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::swap(gpu_device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(data_, other.data_);
    std::swap(extents_, other.extents_);
    std::swap(padding_, other.padding_);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memset(const int pattern, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->memset(pattern, pos, this->size() * sizeof(value_type));
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::fill(const value_type value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->fill(value, pos, this->size());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_device(data_to_copy, 0, this->size());
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size() - pos);
    if (data_to_copy.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->copy_to_device(data_to_copy.data(), pos, rcount);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const_host_pointer_type data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_device(data_to_copy, 0, this->size());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_host(buffer, 0, this->size());
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size() - pos);
    if (buffer.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->copy_to_host(buffer.data(), pos, rcount);
}
template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(host_pointer_type buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_host(buffer, 0, this->size());
}

// explicitly instantiate template class depending on available backends
#if defined(PLSSVM_HAS_CUDA_BACKEND) || defined(PLSSVM_HAS_HIP_BACKEND)
template class gpu_device_ptr<float, int>;
template class gpu_device_ptr<double, int>;
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
template class gpu_device_ptr<float, const ::plssvm::opencl::detail::command_queue *, cl_mem>;
template class gpu_device_ptr<double, const ::plssvm::opencl::detail::command_queue *, cl_mem>;
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
template class gpu_device_ptr<float, ::plssvm::dpcpp::detail::queue>;
template class gpu_device_ptr<double, ::plssvm::dpcpp::detail::queue>;
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
template class gpu_device_ptr<float, ::plssvm::adaptivecpp::detail::queue>;
template class gpu_device_ptr<double, ::plssvm::adaptivecpp::detail::queue>;
    #endif
#endif

}  // namespace plssvm::detail