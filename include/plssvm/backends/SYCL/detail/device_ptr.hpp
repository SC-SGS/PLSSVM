/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Small wrapper around a SYCL device pointer and functions.
 */

#pragma once

// TODO: try to get rid of this include
#include "sycl/sycl.hpp"  // sycl::queue

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::sycl::detail {

/**
 * @brief Returns the number of available (GPU) devices. // TODO:
 * @return the number of devices (`[[nodiscard]]`)
 */
[[nodiscard]] std::size_t get_device_count();
/**
 * @brief Wait for the compute device associated with @p queue to finish.
 * @param[in] queue the SYCL queue to synchronize
 */
void device_synchronize(::sycl::queue &queue);

/**
 * @brief Small wrapper class around a SYCL device pointer together with commonly used device functions.
 * @tparam T the type of the kernel pointer to wrap
 */
template <typename T>
class device_ptr {
  public:
    /// The type of the values used in the SYCL device pointer.
    using value_type = T;
    /// The type of the wrapped SYCL device pointer.
    using pointer = value_type *;
    /// The const type of the wrapped SYCL device pointer.
    using const_pointer = const value_type *;
    /// The used size type.
    using size_type = std::size_t;

    /**
     * @brief Default construct a `device_ptr` with a size of `0`.
     * @details Always associated with device `0`.
     */
    device_ptr() = default;
    /**
     * @brief Allocates `size * sizeof(T)` bytes on the device associated with @p queue.
     * @param[in] size the number of elements represented by the device pointer
     * @param[in] queue the associated SYCL queue
     */
    device_ptr(size_type size, ::sycl::queue &queue);

    /**
     * @brief Move only type, therefore deleted copy-constructor.
     */
    device_ptr(const device_ptr &) = delete;
    /**
     * @brief Move-constructor.
     * @param[inout] other the `device_ptr` to move-construct from
     */
    device_ptr(device_ptr &&other) noexcept;

    /**
     * @brief Move only type, therefore deleted copy-assignment operator.
     */
    device_ptr &operator=(const device_ptr &) = delete;
    /**
     * @brief Move-assignment operator. Uses the copy-and-swap idiom.
     * @param[in] other the `device_ptr` to move-assign from
     * @return `*this`
     */
    device_ptr &operator=(device_ptr &&other) noexcept;

    /**
     * @brief Destruct the device data.
     */
    ~device_ptr();

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[inout] other the other `device_ptr`
     */
    void swap(device_ptr &other) noexcept;
    /**
     * @brief Swap the contents of @p lhs and @p rhs.
     * @param[inout] lhs a `device_ptr`
     * @param[inout] rhs a `device_ptr`
     */
    friend void swap(device_ptr &lhs, device_ptr &rhs) noexcept { lhs.swap(rhs); }

    /**
     * @brief Checks whether `*this` currently wraps a SYCL device pointer.
     * @return `true` if `*this` wraps a device pointer, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != nullptr;
    }
    /**
     * @brief Access the underlying SYCL device pointer.
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
     * @brief Get the number of elements in the wrapped SYCL device pointer.
     * @return the size (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept {
        return size_;
    }
    /**
     * @brief Check whether no elements are currently associated to the SYCL device pointer.
     * @return `true` if no elements are wrapped, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept {
        return size_ == 0;
    }
    /**
     * @brief Return the SYCL queue associated with the wrapped SYCL device pointer.
     * @return the queue (`[[nodiscard]]`)
     */
    [[nodiscard]] ::sycl::queue &queue() const noexcept {
        return *queue_;
    }

    /**
     * @brief Memset all values to @p value starting at position @p pos.
     * @param[in] value the memset value
     * @param[in] pos the position to start the memset
     * @throws plssvm::sycl::backend_exception if @p pos is greater or equal than `device_ptr::size()`
     */
    void memset(value_type value, size_type pos = 0);
    /**
     * @brief Memset up-to @p count values to @p value starting at position @p pos.
     * @details Memset `[p, rcount)` where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] value the memset value
     * @param[in] pos the position to start the memset
     * @param[in] count the number of values to set
     * @throws plssvm::sycl::backend_exception if @p pos is greater or equal than `device_ptr::size()`
     */
    void memset(value_type value, size_type pos, size_type count);

    /**
     * @brief Memcpy `device_ptr::size()` many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     * @throws plssvm::sycl::backend_exception if @p data_to_copy is too small to satisfy the memcpy
     */
    void memcpy_to_device(const std::vector<value_type> &data_to_copy);
    /**
     * @brief Memcpy up-to @p count many values from @p data_to_copy to the device starting at SYCL device pointer position @p pos.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the SYCL device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::sycl::backend_exception if @p data_to_copy is too small to satisfy the memcpy
     */
    void memcpy_to_device(const std::vector<value_type> &data_to_copy, size_type pos, size_type count);
    /**
     * @brief Memcpy `device_ptr::size()` many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     */
    void memcpy_to_device(const_pointer data_to_copy);
    /**
     * @brief Memcpy up-to @p count many values from @p data_to_copy to the device starting at SYCL device pointer position @p pos.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the SYCL device pointer
     * @param[in] count the number of elements to copy
     */
    void memcpy_to_device(const_pointer data_to_copy, size_type pos, size_type count);

    /**
     * @brief Memcpy `device_ptr::size()` many values from the device to the host buffer @p buffer.
     * @param[in] buffer the buffer to copy the data to
     * @throws plssvm::sycl::backend_exception if @p buffer is too small
     */
    void memcpy_to_host(std::vector<value_type> &buffer);
    /**
     * @brief Memcpy up-to @p count many values from the device starting at SYCL device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the SYCL device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::sycl::backend_exception if @p data_to_copy is too small
     */
    void memcpy_to_host(std::vector<value_type> &buffer, size_type pos, size_type count);
    /**
     * @brief Memcpy `device_ptr::size()` many values from the device to the host buffer @p buffer.
     * @param[in] buffer the buffer to copy the data to
     */
    void memcpy_to_host(pointer buffer);
    /**
     * @brief Memcpy up-to @p count many values from the device starting at SYCL device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[p, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the SYCL device pointer
     * @param[in] count the number of elements to copy
     */
    void memcpy_to_host(pointer buffer, size_type pos, size_type count);

  private:
    ::sycl::queue *queue_ = nullptr;
    pointer data_ = nullptr;
    size_type size_ = 0;
};

extern template class device_ptr<float>;
extern template class device_ptr<double>;

}  // namespace plssvm::sycl::detail