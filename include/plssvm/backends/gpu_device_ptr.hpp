/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends using a GPU. Used for code duplication reduction.
 */

#ifndef PLSSVM_BACKENDS_GPU_DEVICE_PTR_HPP_
#define PLSSVM_BACKENDS_GPU_DEVICE_PTR_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/type_list.hpp"       // plssvm::detail::{supported_real_types, tuple_contains_v}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::gpu_device_ptr_exception
#include "plssvm/matrix.hpp"                 // plssvm::layout_type, plssvm::matrix

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief Small wrapper class around a GPU device pointer together with commonly used device functions for all GPU backends to reduce code duplication.
 * @tparam T the type of the data
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 * @tparam device_pointer_t the type of the device pointer (dependent on the used backend; default: `T*`)
 */
template <typename T, typename queue_t, typename device_pointer_t = T *>
class gpu_device_ptr {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<T, detail::supported_real_types>, "Illegal real type provided! See the 'real_type_list' in the type_list.hpp header for a list of the allowed types.");

  public:
    /// The type of the values used in the device_ptr.
    using value_type = T;
    /// The type of the host pointer corresponding to the wrapped device pointer.
    using host_pointer_type = value_type *;
    /// The const type of the host pointer corresponding to the wrapped device pointer.
    using const_host_pointer_type = const value_type *;
    /// The used size type.
    using size_type = std::size_t;
    /// The type of the device queue used to manipulate the managed device memory.
    using queue_type = queue_t;
    /// The type of the device pointer.
    using device_pointer_type = device_pointer_t;

    /**
     * @brief Default construct a gpu_device_ptr with a size of 0.
     */
    gpu_device_ptr() = default;
    /**
     * @brief Construct a device_ptr for the device managed by @p queue with the extents { @p size, 1 }.
     * @param[in] size the size of the managed memory
     * @param[in] queue the queue (or similar) to manage the device_ptr
     */
    gpu_device_ptr(size_type size, const queue_type queue);
    /**
     * @brief Construct a device_ptr for the device managed by @p queue with the extents @p extents.
     * @details The managed memory size is: extents[0] * extents[1].
     * @param[in] extents the extents of the managed memory; size = extents[0] * extents[1]
     * @param[in] queue the queue (or similar) to manage the device_ptr
     */
    gpu_device_ptr(std::array<size_type, 2> extents, const queue_type queue);
    /**
     * @brief Construct a device_ptr for the device managed by @p queue with the extents @p extents including @p padding.
     * @details The managed memory size is: (extents[0] + padding[0]) * (extents[1] + padding[1]).
     * @param[in] extents the extents of the managed memory
     * @param[in] padding the padding applied to the extents
     * @param[in] queue the queue (or similar) to manage the device_ptr
     */
    gpu_device_ptr(std::array<size_type, 2> extents, std::array<size_type, 2> padding, const queue_type queue);

    /**
     * @brief Delete copy-constructor to make device_ptr a move only type.
     */
    gpu_device_ptr(const gpu_device_ptr &) = delete;
    /**
     * @brief Move-constructor as device_ptr is a move-only type.
     * @param[in,out] other the device_ptr to move-construct from
     */
    gpu_device_ptr(gpu_device_ptr &&other) noexcept;

    /**
     * @brief Delete copy-assignment-operator to make device_ptr a move only type.
     */
    gpu_device_ptr &operator=(const gpu_device_ptr &) = delete;
    /**
     * @brief Move-assignment-operator as device_ptr is a move-only type.
     * @param[in,out] other the device_ptr to move-assign from
     * @return `*this`
     */
    gpu_device_ptr &operator=(gpu_device_ptr &&other) noexcept;

    /**
     * @brief Free the memory managed by the device_ptr.
     */
    virtual ~gpu_device_ptr() = default;

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other device_ptr
     */
    void swap(gpu_device_ptr &other) noexcept;
    /**
     * @brief Swap the contents of @p lhs and @p rhs.
     * @param[in,out] lhs a device_ptr
     * @param[in,out] rhs a device_ptr
     */
    friend void swap(gpu_device_ptr &lhs, gpu_device_ptr &rhs) noexcept { lhs.swap(rhs); }

    /**
     * @brief Checks whether `*this` currently wraps a device pointer.
     * @details Same as `device_ptr::get() != nullptr`.
     * @return `true` if `*this` wraps a device pointer, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] explicit operator bool() const noexcept {
        return data_ != device_pointer_type{};
    }
    /**
     * @brief Access the underlying device pointer.
     * @return the device pointer (`[[nodiscard]]`)
     */
    [[nodiscard]] device_pointer_type get() noexcept {
        return data_;
    }
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::get()
     */
    [[nodiscard]] device_pointer_type get() const noexcept {
        return data_;
    }
    /**
     * @brief Get the number of elements in the wrapped device_ptr.
     * @details Same as: `this->size(0) * this->size(1)`.
     * @return the number of elements (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept {
        return (extents_[0] + padding_[0]) * std::max(std::size_t{ 1 }, extents_[1] + padding_[1]);
    }
    /**
     * @brief Get the number of elements in the @p extent direction in the wrapped device_ptr.
     * @param[in] extent the extent to retrieve
     * @return the number of elements in direction @p extent (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size(const size_type extent) const noexcept {
        PLSSVM_ASSERT(extent < 2, "Only extents 0 and 1 are allowed, but {} was provided!", extent);
        return extents_[extent];
    }
    /**
     * @brief Get the number of elements in both directions in the wrapped device_ptr.
     * @return the number of elements in both directions (`[[nodiscard]]`)
     */
    [[nodiscard]] std::array<size_type, 2> extents() const noexcept {
        return extents_;
    }
    [[nodiscard]] size_type padding(const size_type pad) const noexcept {
        PLSSVM_ASSERT(pad < 2, "Only extents 0 and 1 are allowed, but {} was provided!", pad);
        return padding_[pad];
    }
    /**
     * @brief Check whether the device_ptr currently maps zero elements.
     * @details Same as `device_ptr::size() == 0`.
     * @return `true` if no elements are wrapped, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept {
        return this->size() == 0;
    }
    /**
     * @brief Return the queue managing the memory of the wrapped device pointer.
     * @return the device queue (`[[nodiscard]]`)
     */
    [[nodiscard]] queue_type queue() const noexcept {
        return queue_;
    }

    /**
     * @brief Memset all bytes using the @p pattern starting at position @p pos.
     * @param[in] pattern the memset pattern
     * @param[in] pos the position to start the memset operation
     * @throws plssvm::gpu_device_ptr_exception if @p pos is greater or equal than device_ptr::size()
     */
    void memset(int pattern, size_type pos = 0);
    /**
     * @brief Memset up-to @p num_bytes values to @p pattern starting at position @p pos.
     * @details Memset `[pos, rnum_bytes)` where `num_bytes` is the smaller value of @p num_bytes and `(device_ptr::size() - pos) * sizeof(value_type)`.
     * @param[in] pattern the memset value
     * @param[in] pos the position to start the memset
     * @param[in] num_bytes the number of bytes to set
     * @throws plssvm::gpu_device_ptr_exception if @p pos is greater or equal than device_ptr::size()
     */
    virtual void memset(int pattern, size_type pos, size_type num_bytes) = 0;

    /**
     * @brief Fill all values with the @p value starting at position @p pos.
     * @param[in] value the fill value
     * @param[in] pos the position to start the fill
     * @throws plssvm::gpu_device_ptr_exception if @p pos is greater or equal than device_ptr::size()
     */
    void fill(value_type value, size_type pos = 0);
    /**
     * @brief Fill up-to @p count values to @p value starting at position @p pos.
     * @details Fill `[pos, rcount)` where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] value the fill value
     * @param[in] pos the position to start the fill
     * @param[in] count the number of values to set
     * @throws plssvm::gpu_device_ptr_exception if @p pos is greater or equal than device_ptr::size()
     */
    virtual void fill(value_type value, size_type pos, size_type count) = 0;

    /**
     * @brief Copy device_ptr::size() many values from @p data_to_copy to the device.
     * @tparam layout the layout type of the matrix
     * @param[in] data_to_copy the data to copy onto the device
     * @throws plssvm::gpu_device_ptr_exception if @p data_to_copy is too small to satisfy the copy
     */
    template <layout_type layout>
    void copy_to_device(const matrix<value_type, layout> &data_to_copy) {
        PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

        if (data_to_copy.num_entries_padded() < this->size()) {
            throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", this->size(), data_to_copy.num_entries_padded()) };
        }
        this->copy_to_device(data_to_copy.data());
    }
    /**
     * @brief Copy device_ptr::size() many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     * @throws plssvm::gpu_device_ptr_exception if @p data_to_copy is too small to satisfy the copy
     */
    void copy_to_device(const std::vector<value_type> &data_to_copy);
    /**
     * @brief Copy up-to @p count many values from @p data_to_copy to the device starting at device pointer position @p pos.
     * @details Copies `[pos, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::gpu_device_ptr_exception if @p data_to_copy is too small to satisfy the copy
     */
    void copy_to_device(const std::vector<value_type> &data_to_copy, size_type pos, size_type count);
    /**
     * @brief Copy device_ptr::size() many values from @p data_to_copy to the device.
     * @param[in] data_to_copy the data to copy onto the device
     */
    void copy_to_device(const_host_pointer_type data_to_copy);
    /**
     * @brief Copy up-to @p count many values from @p data_to_copy to the device starting at device pointer position @p pos.
     * @details Copies `[pos, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] pos the starting position for the copying in the device pointer
     * @param[in] count the number of elements to copy
     */
    virtual void copy_to_device(const_host_pointer_type data_to_copy, size_type pos, size_type count) = 0;

    /**
     * @brief Copy device_ptr::size() many values from the device to the host buffer @p buffer.
     * @tparam layout the layout type of the matrix
     * @param[in] buffer the buffer to copy the data to
     * @throws plssvm::gpu_device_ptr_exception if @p buffer is too small to satisfy the copy
     */
    template <layout_type layout>
    void copy_to_host(matrix<value_type, layout> &buffer) const {
        PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

        if (buffer.num_entries_padded() < this->size()) {
            throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", this->size(), buffer.num_entries_padded()) };
        }
        this->copy_to_host(buffer.data());
    }
    /**
     * @brief Copy device_ptr::size() many values from the device to the host buffer @p buffer.
     * @param[out] buffer the buffer to copy the data to
     * @throws plssvm::gpu_device_ptr_exception if @p buffer is too small to satisfy the copy
     */
    void copy_to_host(std::vector<value_type> &buffer) const;
    /**
     * @brief Copy up-to @p count many values from the device starting at device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[pos, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[out] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the device pointer
     * @param[in] count the number of elements to copy
     * @throws plssvm::gpu_device_ptr_exception if @p data_to_copy is too small to satisfy the copy
     */
    void copy_to_host(std::vector<value_type> &buffer, size_type pos, size_type count) const;
    /**
     * @brief Copy device_ptr::size() many values from the device to the host buffer @p buffer.
     * @param[out] buffer the buffer to copy the data to
     */
    void copy_to_host(host_pointer_type buffer) const;
    /**
     * @brief Copy up-to @p count many values from the device starting at device pointer position @p pos to the host buffer @p buffer.
     * @details Copies `[pos, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param[out] buffer the buffer to copy the data to
     * @param[in] pos the starting position for the copying in the device pointer
     * @param[in] count the number of elements to copy
     */
    virtual void copy_to_host(host_pointer_type buffer, size_type pos, size_type count) const = 0;

  protected:
    /// The device queue used to manage the device memory associated with this device pointer.
    queue_type queue_{};
    /// The device pointer pointing to the managed memory.
    device_pointer_type data_{};
    /// The size of the managed memory.
    std::array<size_type, 2> extents_{ { 0, 0 } };
    /// The padding size of the managed memory.
    std::array<size_type, 2> padding_{ { 0, 0 } };
};

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_DEVICE_PTR_HPP_