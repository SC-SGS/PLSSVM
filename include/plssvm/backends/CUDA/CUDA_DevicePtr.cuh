/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Small wrapper around a CUDA device pointer and functions.
 */

#pragma once

#include "plssvm/backends/CUDA/CUDA_exceptions.hpp"  // plssvm::cuda_backend_exception

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <utility>    // std::move, std::swap, std::exchange

#define PLSSVM_CUDA_ERROR_CHECK(err) plssvm::detail::cuda::gpu_assert((err));

// TODO: correct names (including doxygen comments), correct file, excpetion file?
// TODO: correct namespace
namespace plssvm::detail::cuda {

/**
 * @brief Check the CUDA error code. If @p code signals an error, throw a `plssvm::cuda_backend_exception`.
 * @details The exception contains the error name and error string for more debug information.
 * @param[in] code the CUDA error code to check
 * @throws plssvm::cuda_backend_exception if the error code signals a failure
 */
inline void gpu_assert(const cudaError_t code) {
    if (code != cudaSuccess) {
        throw cuda_backend_exception{ fmt::format("CUDA assert {}: {}", cudaGetErrorName(code), cudaGetErrorString(code)) };
    }
}

/**
 * @brief Returns the number of available devices.
 * @return the number of devices (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::size_t get_device_count() {
    int count;
    PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));
    return static_cast<std::size_t>(count);
}
/**
 * @brief Wait for the current compute device to finish.
 * @details Calls `peek_at_last_error()` before synchronizing.
 */
inline void device_synchronize() {
    peek_at_last_error();
    PLSSVM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
/**
 * @brief Wait for the compute device @p device to finish.
 * @details Calls `peek_at_last_error()` before synchronizing.
 * @param[in] device the CUDA device to synchronize
 * @throws plssvm::cuda_backend_exception if the given device ID is smaller than `0` or greater or equal than the available number of devices
 */
inline void device_synchronize(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw plssvm::cuda_backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), device) };
    }
    peek_at_last_error();
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device));
    PLSSVM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}
/**
 * @brief Returns the last error from a runtime call.
 */
inline void peek_at_last_error() {
    PLSSVM_CUDA_ERROR_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Small wrapper class around a CUDA device pointer together with commonly used device functions.
 * @tparam T the type of the kernel pointer to wrap
 */
template <typename T>
class device_ptr {
  public:
    /// The type of the values used in the CUDA device pointer.
    using value_type = T;
    /// The type of the wrapped CUDA device pointer.
    using pointer = value_type *;
    /// The const type of the wrapped CUDA device pointer.
    using const_pointer = const value_type *;
    /// The used size type.
    using size_type = std::size_t;

    /**
     * @brief Default construct a `device_ptr` with a size of `0`.
     * @details Always associated with device `0`.
     */
    device_ptr() = default;
    /**
     * @brief Allocates `size * sizeof(T)` bytes on the device with ID @p device.
     * @param[in] size the number of elements represented by the device pointer
     * @param[in] device the associated CUDA device (default: `0`)
     * @throws plssvm::cuda_backend_exception if the given device ID is smaller than `0` or greater or equal than the available number of devices
     */
    device_ptr(const size_type size, const int device = 0) :
        device_{ device }, size_{ size } {
        if (device_ < 0 || device_ >= static_cast<int>(get_device_count())) {
            throw plssvm::cuda_backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), device_) };
        }
        PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
        PLSSVM_CUDA_ERROR_CHECK(cudaMalloc(&data_, size_ * sizeof(value_type)));
    }

    /**
     * @brief Move only type, therefore deleted copy-constructor.
     */
    device_ptr(const device_ptr &) = delete;
    /**
     * @brief Move-constructor.
     * @param[inout] other the `device_ptr` to move-construct from
     */
    device_ptr(device_ptr &&other) noexcept :
        device_{ std::exchange(other.device_, 0) },
        data_{ std::exchange(other.data_, nullptr) },
        size_{ std::exchange(other.size_, 0) } {}

    /**
     * @brief Move only type, therefore deleted copy-assignment operator.
     */
    device_ptr &operator=(const device_ptr &) = delete;
    /**
     * @brief Move-assignment operator. Uses the copy-and-swap idiom.
     * @param[in] other the `device_ptr` to move-assign from
     * @return `*this`
     */
    device_ptr &operator=(device_ptr &&other) noexcept {
        device_ptr tmp{ std::move(other) };
        this->swap(tmp);
        return *this;
    }

    /**
     * @brief Destruct the device data.
     */
    ~device_ptr() {
        PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
        PLSSVM_CUDA_ERROR_CHECK(cudaFree(data_));
    }

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[inout] other the other `device_ptr`
     */
    void swap(device_ptr &other) noexcept {
        std::swap(device_, other.device_);
        std::swap(data_, other.data_);
        std::swap(size_, other.size_);
    }
    /**
     * @brief Swap the contents of @p lhs and @p rhs.
     * @param[inout] lhs a `device_ptr`
     * @param[inout] rhs a `device_ptr`
     */
    friend void swap(device_ptr &lhs, device_ptr &rhs) noexcept {
        lhs.swap(rhs);
    }

    /**
     * @brief Checks whether `*this` currently wraps a CUDA device pointer.
     * @return `true` if `*this` wraps a device pointer, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr;
    }
    /**
     * @brief Access the underlying CUDA device pointer.
     * @return the device pointer (`[[nodiscard]]`)
     */
    [[nodiscard]] pointer get() noexcept {
        return data_;
    }
    /**
     * @copydoc device_ptr::get()
     */
    [[nodiscard]] const_pointer get() const noexcept {
        return data_;
    }
    /**
     * @brief Get the number of elements in the wrapped CUDA device pointer.
     * @return the size (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }
    /**
     * @brief Check whether no elements are currently associated to the CUDA device pointer.
     * @return `true` if no elements are wrapped, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }
    /**
     * @brief Return the device associated with the wrapped CUDA device pointer.
     * @return the device ID (`[[nodiscard]]`)
     */
    [[nodiscard]] int device() const noexcept {
        return device_;
    }

    /**
     * @brief Memset all values to @p value starting at position @p pos.
     * @param[in] value the memset value
     * @param[in] pos the position to start the memset (default: `0`)
     * @throws plssvm::cuda_backend_exception if @p pos is greater or equal than `device_ptr::size()`
     */
    void memset(const value_type value, const size_type pos = 0) {
        this->memset(value, pos, size_);
    }
    /**
     * @brief Memset up-to @p count values to @p value starting at position @p pos.
     * @details Memset `[p, rcount)` where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] value the memset value
     * @param[in] pos the position to start the memset (default: `0`)
     * @param[in] count the number of values to set
     * @throws plssvm::cuda_backend_exception if @p pos is greater or equal than `device_ptr::size()`
     */
    void memset(const value_type value, const size_type pos, const size_type count) {
        if (pos >= size_) {
            throw plssvm::cuda_backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
        }
        PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
        const size_type rcount = std::min(count, size_ - pos);
        PLSSVM_CUDA_ERROR_CHECK(cudaMemset(data_ + pos, value, rcount * sizeof(value_type)));
    }

    /**
     * @brief Memcpy `device_ptr::size()` many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     * @throws plssvm::cuda_backend_exception if @p data_to_copy is too small to satisfy the memcpy
     */
    void memcpy_to_device(const std::vector<value_type> &data_to_copy) {
        this->memcpy_to_device(data_to_copy, 0, size_);
    }
    /**
     * @brief Memcpy up-to @p count many values from @p data_to_copy to the device starting at CUDA device pointer position @p pos.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the CUDA device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::cuda_backend_exception if @p data_to_copy is too small to satisfy the memcpy
     */
    void memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
        const size_type rcount = std::min(count, size_ - pos);
        if (data_to_copy.size() < rcount) {
            throw plssvm::cuda_backend_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
        }
        this->memcpy_to_device(data_to_copy.data(), pos, count);
    }
    /**
     * @brief Memcpy `device_ptr::size()` many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     */
    void memcpy_to_device(const_pointer data_to_copy) {
        this->memcpy_to_device(data_to_copy, 0, size_);
    }
    /**
     * @brief Memcpy up-to @p count many values from @p data_to_copy to the device starting at CUDA device pointer position @p pos.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the CUDA device pointer
     * @param[in] count the number of elements to copy
     */
    void memcpy_to_device(const_pointer data_to_copy, const size_type pos, const size_type count) {
        PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
        const size_type rcount = std::min(count, size_ - pos);
        PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), cudaMemcpyHostToDevice));
    }

    /**
     * @brief Memcpy `device_ptr::size()` many values from the device to the host buffer @p buffer.
     * @param[in] buffer the buffer to copy the data to
     * @throws plssvm::cuda_backend_exception if @p buffer is too small
     */
    void memcpy_to_host(std::vector<value_type> &buffer) {
        this->memcpy_to_host(buffer, 0, size_);
    }
    /**
     * @brief Memcpy up-to @p count many values from the device starting at CUDA device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the CUDA device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::cuda_backend_exception if @p data_to_copy is too small
     */
    void memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) {
        const size_type rcount = std::min(count, size_ - pos);
        if (buffer.size() < rcount) {
            throw plssvm::cuda_backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
        }
        this->memcpy_to_host(buffer.data(), pos, size_);
    }
    /**
     * @brief Memcpy `device_ptr::size()` many values from the device to the host buffer @p buffer.
     * @param[in] buffer the buffer to copy the data to
     */
    void memcpy_to_host(pointer buffer) {
        this->memcpy_to_host(buffer, 0, size_);
    }
    /**
     * @brief Memcpy up-to @p count many values from the device starting at CUDA device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the CUDA device pointer
     * @param[in] count the number of elements to copy
     */
    void memcpy_to_host(pointer buffer, const size_type pos, const size_type count) {
        PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device_));
        const size_type rcount = std::min(count, size_ - pos);
        PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), cudaMemcpyDeviceToHost));
    }

  private:
    int device_ = 0;
    pointer data_ = nullptr;
    size_type size_ = 0;
};

#undef PLSSVM_CUDA_ERROR_CHECK

}  // namespace plssvm::detail::cuda