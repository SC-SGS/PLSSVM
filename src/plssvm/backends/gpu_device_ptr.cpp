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
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue

    #include "CL/cl.h"  // cl_mem
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
#include "plssvm/shape.hpp"                  // plssvm::shape

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <array>      // std::array
#include <memory>     // std::addressof
#include <utility>    // std::exchange, std::swap
#include <vector>     // std::vector

namespace plssvm::detail {

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(const size_type size, const queue_type queue) :
    queue_{ queue },
    shape_{ plssvm::shape{ size, 1 } } { }

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(const plssvm::shape shape, const queue_type queue) :
    queue_{ queue },
    shape_{ shape } { }

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(const plssvm::shape shape, const plssvm::shape padding, const queue_type queue) :
    queue_{ queue },
    shape_{ shape },
    padding_{ padding } { }

template <typename T, typename queue_t, typename device_pointer_t>
gpu_device_ptr<T, queue_t, device_pointer_t>::gpu_device_ptr(gpu_device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, queue_type{}) },
    shape_{ std::exchange(other.shape_, plssvm::shape{}) },
    padding_{ std::exchange(other.padding_, plssvm::shape{}) },
    data_{ std::exchange(other.data_, device_pointer_type{}) } { }

template <typename T, typename queue_t, typename device_pointer_t>
auto gpu_device_ptr<T, queue_t, device_pointer_t>::operator=(gpu_device_ptr &&other) noexcept -> gpu_device_ptr & {
    // guard against self-assignment
    if (this != std::addressof(other)) {
        queue_ = std::exchange(other.queue_, queue_type{});
        shape_ = std::exchange(other.shape_, plssvm::shape{});
        padding_ = std::exchange(other.padding_, plssvm::shape{});
        data_ = std::exchange(other.data_, device_pointer_type{});
    }
    return *this;
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::swap(gpu_device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(shape_, other.shape_);
    std::swap(padding_, other.padding_);
    std::swap(data_, other.data_);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::memset(const int pattern, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->memset(pattern, pos, this->size_padded() * sizeof(value_type));
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::fill(const value_type value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->fill(value, pos, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_device(data_to_copy, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (data_to_copy.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->copy_to_device(data_to_copy.data(), pos, rcount);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device(const_host_pointer_type data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_device(data_to_copy, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_device_strided(const std::vector<value_type> &data_to_copy, std::size_t spitch, std::size_t width, std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(width <= spitch, "Invalid width and spitch combination specified!");

    if (width > spitch) {
        throw gpu_device_ptr_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }
    if (width * height > data_to_copy.size()) {
        throw gpu_device_ptr_exception{ fmt::format("The sub-matrix ({}x{}) to copy is to big ({})!", width, height, data_to_copy.size()) };
    }

    this->copy_to_device_strided(data_to_copy.data(), spitch, width, height);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_host(buffer, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (buffer.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->copy_to_host(buffer.data(), pos, rcount);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_host(host_pointer_type buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_host(buffer, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_other_device(gpu_device_ptr<T, queue_t, device_pointer_t> &target) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.data_ != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    this->copy_to_other_device(target, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_other_device(gpu_device_ptr<T, queue_t, device_pointer_t> &target, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.data_ != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }
    this->copy_to_other_device(target.get(), pos, rcount);
}

template <typename T, typename queue_t, typename device_pointer_t>
void gpu_device_ptr<T, queue_t, device_pointer_t>::copy_to_other_device(device_pointer_type target) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    this->copy_to_other_device(target, 0, this->size_padded());
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
