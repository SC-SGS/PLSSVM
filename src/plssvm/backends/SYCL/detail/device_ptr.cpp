/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"

#include "plssvm/backends/SYCL/exceptions.hpp"  // plssvm::sycl::backend_exception

#include "fmt/core.h"     // fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::platform, sycl::gpu_selector

#include <cstddef>  // std::size_t
#include <utility>  // std::exchange, std::move, std::swap
#include <vector>   // std::vector

namespace plssvm::sycl::detail {

[[nodiscard]] std::size_t get_device_count() {
    // TODO:
    return ::sycl::platform(::sycl::gpu_selector{}).get_devices().size();
}
void device_synchronize(::sycl::queue &queue) {
    queue.wait_and_throw();
}

template <typename T>
device_ptr<T>::device_ptr(const size_type size, ::sycl::queue &queue) :
    queue_{ &queue }, size_{ size } {
    data_ = ::sycl::malloc_device<value_type>(size_, *queue_);
}

template <typename T>
device_ptr<T>::device_ptr(device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, nullptr) },
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
    if (queue_ != nullptr) {
        ::sycl::free(reinterpret_cast<void *>(data_), *queue_);
    }
}

template <typename T>
void device_ptr<T>::swap(device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

template <typename T>
void device_ptr<T>::memset(const value_type value, const size_type pos) {
    this->memset(value, pos, size_);
}
template <typename T>
void device_ptr<T>::memset(const value_type value, const size_type pos, const size_type count) {
    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rcount = std::min(count, size_ - pos);
    queue_->memset(reinterpret_cast<void *>(data_ + pos), value, rcount * sizeof(value_type)).wait();
}

template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy) {
    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    const size_type rcount = std::min(count, size_ - pos);
    if (data_to_copy.size() < rcount) {
        throw backend_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->memcpy_to_device(data_to_copy.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy) {
    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy, const size_type pos, const size_type count) {
    const size_type rcount = std::min(count, size_ - pos);
    queue_->copy(data_to_copy, data_, rcount).wait();
    //    queue_->memcpy(data_, data_to_copy, rcount * sizeof(value_type)).wait();
}

template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer) {
    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) {
    const size_type rcount = std::min(count, size_ - pos);
    if (buffer.size() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->memcpy_to_host(buffer.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer) {
    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer, const size_type pos, const size_type count) {
    const size_type rcount = std::min(count, size_ - pos);
    queue_->copy(data_, buffer, rcount).wait();
    //    queue_->memcpy(reinterpret_cast<void *>(buffer), const_cast<const void *>(reinterpret_cast<void *>(data_)), rcount * sizeof(value_type)).wait();
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::sycl::detail