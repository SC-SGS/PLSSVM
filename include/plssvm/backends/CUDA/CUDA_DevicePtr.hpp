#pragma once

#include "CUDA_exceptions.hpp"

#include "fmt/core.h"

#include <algorithm>  // std::min
#include <utility>    // std::move, std::swap, std::exchange

// TODO: correct namespace
namespace plssvm::detail::cuda {

std::size_t get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return static_cast<std::size_t>(count);
}

void device_synchronize() {
    cudaDeviceSynchronize();
}
void peek_at_last_error() {
    cudaPeekAtLastError();
}

template <typename T>
class device_ptr {
  public:
    using value_type = T;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using reference = value_type &;
    using const_reference = const value_type &;
    using size_type = std::size_t;

    device_ptr() = default;
    device_ptr(const int device, const size_type size) :
        device_{ device }, size_{ size } {
        if (device_ < 0) {
            throw plssvm::cuda_backend_exception{ fmt::format("Device ID must not be negative but is {}!", device_) };
        }
        cudaSetDevice(device_);
        cudaMalloc(static_cast<void **>(data_), size_ * sizeof(value_type));
    }

    device_ptr(const device_ptr &) = delete;
    device_ptr(device_ptr &&other) noexcept :
        device_{ std::exchange(other.device_, 0) },
        data_{ std::exchnage(other.data_, nullptr) },
        size_{ std::exchange(other.size_, 0) } {}

    device_ptr &operator=(const device_ptr &) = delete;
    device_ptr &operator=(device_ptr &&other) noexcept {
        device_ptr tmp{ std::move(other) };
        this->swap(tmp);
        return *this;
    }

    ~device_ptr() {
        cudaSetDevice(device_);
        cudaFree(data_);
    }

    void swap(device_ptr &other) noexcept {
        std::swap(device_, other.device_);
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    friend void swap(device_ptr &lhs, device_ptr &rhs) noexcept {
        lhs.swap(rhs);
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

    [[nodiscard]] pointer get() noexcept {
        return data_;
    }
    [[nodiscard]] const_pointer get() const noexcept {
        return data_;
    }
    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }
    [[nodiscard]] int device() const noexcept {
        return device_;
    }

    void memset(const value_type value, const size_type pos = 0, const size_type count = size_) {
        if (pos >= size_) {
            throw cuda_backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
        }
        cudaSetDevice(device_);
        const size_type rcount = std::min(count, size_ - pos);
        cudaMemset(data_ + pos, value, rcount * sizeof(value_type));
    }

    void memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos = 0, const size_type count = size_) {
        const size_type rcount = std::min(count, size_ - pos);
        if (data_to_copy.size() < rcount) {
            throw cuda_backend_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
        }
        this->memcpy_to_device(data_to_copy.data(), pos, count);
    }
    void memcpy_to_device(const_pointer data_to_copy, const size_type pos = 0, const size_type count = size_) {
        cudaSetDevice(device_);
        const size_type rcount = std::min(count, size_ - pos);
        cudaMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), cudaMemcpyHostToDevice);
    }

    void memcpy_to_host(std::vector<value_type> &buffer, const size_type pos = 0, const size_type count = size_) {
        const size_type rcount = std::min(count, size_ - pos);
        if (buffer.size() < rcount) {
            throw cuda_backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
        }
        this->memcpy_to_host(buffer.data(), pos, size);
    }
    void mempcy_to_host(pointer buffer, const size_type pos = 0, const size_type count = size_) {
        cudaSetDevice(device_);
        const size_type rcount = std::min(count, size_ - pos);
        cudaMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), cudaMemcpyDeviceToHost);
    }

  private:
    int device_ = 0;
    pointer data_ = nullptr;
    size_type size_ = 0;
};

}  // namespace plssvm::detail::cuda