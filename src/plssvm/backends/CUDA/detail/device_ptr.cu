/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"
#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <utility>    // std::exchange, std::move, std::swap
#include <vector>     // std::vector

#define PLSSVM_CUDA_ERROR_CHECK(err) plssvm::cuda::detail::gpu_assert((err))

namespace plssvm::cuda::detail {

/**
 * @brief Check the CUDA error code. If @p code signals an error, throw a `plssvm::cuda_backend_exception`.
 * @details The exception contains the error name and error string for more debug information.
 * @param[in] code the CUDA error code to check
 * @throws plssvm::cuda_backend_exception if the error code signals a failure
 */
inline void gpu_assert(const cudaError_t code) {
    if (code != cudaSuccess) {
        throw backend_exception{ fmt::format("CUDA assert {}: {}", cudaGetErrorName(code), cudaGetErrorString(code)) };
    }
}

int get_device_count() {
    int count;
    PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));
    return count;
}
void set_device(const int device) {
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device));
}

void peek_at_last_error() {
    PLSSVM_CUDA_ERROR_CHECK(cudaPeekAtLastError());
}
void device_synchronize() {
    peek_at_last_error();
    PLSSVM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
void device_synchronize(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), device) };
    }
    peek_at_last_error();
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device));
    PLSSVM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const int device) :
    device_{ device }, size_{ size } {
    if (device_ < 0 || device_ >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), device_) };
    }
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
    PLSSVM_CUDA_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type)));
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
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
    PLSSVM_CUDA_ERROR_CHECK(cudaFree(data_));
}

template <typename T>
void device_ptr<T>::swap(device_ptr &other) noexcept {
    std::swap(device_, other.device_);
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
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_CUDA_ERROR_CHECK(cudaMemset(data_ + pos, value, rcount * sizeof(value_type)));
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
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), cudaMemcpyHostToDevice));
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
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), cudaMemcpyDeviceToHost));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::cuda::detail