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
#include "plssvm/shape.hpp"                  // plssvm::shape

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief Small wrapper class around a GPU device pointer together with commonly used device functions for all GPU backends to reduce code duplication.
 * @tparam derived_gpu_device_ptr the type of the derived device ptr (using CRTP)
 */
template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
class gpu_device_ptr {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<T, detail::supported_real_types>,
                  "Illegal real type provided! See the 'real_type_list' in the type_list.hpp header for a list of the allowed types.");

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
     * @param[in] use_usm_allocations if `true` use USM allocations in the respective backend
     */
    gpu_device_ptr(size_type size, const queue_type queue, bool use_usm_allocations);
    /**
     * @brief Construct a device_ptr for the device managed by @p queue with the provided @p shape.
     * @details The managed memory size is: extents[0] * extents[1].
     * @param[in] shape the 2D size of the managed memory; size = shape.x * shape.y
     * @param[in] queue the queue (or similar) to manage the device_ptr
     * @param[in] use_usm_allocations if `true` use USM allocations in the respective backend
     */
    gpu_device_ptr(plssvm::shape shape, const queue_type queue, bool use_usm_allocations);
    /**
     * @brief Construct a device_ptr for the device managed by @p queue with the provided @p shape including @p padding.
     * @details The managed memory size is: (shape.x + padding.x) * (shape.y + padding.y).
     * @param[in] shape the extents of the managed memory
     * @param[in] padding the padding applied to the extents
     * @param[in] queue the queue (or similar) to manage the device_ptr
     * @param[in] use_usm_allocations if `true` use USM allocations in the respective backend
     */
    gpu_device_ptr(plssvm::shape shape, plssvm::shape padding, const queue_type queue, bool use_usm_allocations);

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
        return shape_.x * shape_.y;
    }

    /**
     * @brief Get the number of elements in both dimensions in the wrapped device_ptr.
     * @return the number of elements in both directions (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape shape() const noexcept {
        return shape_;
    }

    /**
     * @brief Check whether the device_ptr currently maps zero elements.
     * @details Same as `device_ptr::size() == 0`.
     * @return `true` if no elements are wrapped, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept {
        return shape_.x == 0 && shape_.y == 0;
    }

    /**
     * @brief Get the number of padding entries in both dimensions in the wrapped device_ptr.
     * @return the number of padding entries in both directions (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape padding() const noexcept {
        return padding_;
    }

    /**
     * @brief Get the number of values **including** padding in the wrapped device_ptr.
     * @return the number of elements (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size_padded() const noexcept {
        return (shape_.x + padding_.x) * (shape_.y + padding_.y);
    }

    /**
     * @brief Get the number of values in both dimensions **including** padding in the wrapped device_ptr.
     * @return the number of elements in both directions (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape shape_padded() const noexcept {
        return plssvm::shape{ shape_.x + padding_.x, shape_.y + padding_.y };
    }

    /**
     * @brief Checks whether the wrapped device_ptr contains any padding entries.
     * @return `true` if the wrapped device_ptr is padded, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_padded() const noexcept {
        return !(padding_.x == 0 && padding_.y == 0);
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
    void copy_to_device(const matrix<value_type, layout> &data_to_copy);

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
     * @brief Copy the sub-matrix starting at @p row with @p num_rows to the device.
     * @details If the @p layout is AoS, uses a simple linear copy to the device (no strides needed)
     * @note The device_ptr must be constructed with an two-dimensional shape in order to use this function!
     * @tparam layout the layout type of the matrix
     * @param[in] data_to_copy the data to copy the sub-matrix from onto the device
     * @param[in] start_row the first row of the sub-matrix
     * @param[in] num_rows the number of rows in the sub-matrix
     */
    template <layout_type layout>
    void copy_to_device_strided(const matrix<value_type, layout> &data_to_copy, std::size_t start_row, std::size_t num_rows);

    /**
     * @brief Copy a matrix (@p height rows of @p width) from @p data_to_copy to the device.
     * @note The device_ptr must be constructed with an two-dimensional shape in order to use this function!
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] spitch the stride length
     * @param[in] width the width of the 2D matrix to copy
     * @param[in] height the height of the 2D matrix to copy
     */
    void copy_to_device_strided(const std::vector<value_type> &data_to_copy, std::size_t spitch, std::size_t width, std::size_t height);

    /**
     * @brief Copy a matrix (@p height rows of @p width) from @p data_to_copy to the device.
     * @note The device_ptr must be constructed with an two-dimensional shape in order to use this function!
     * @param[in] data_to_copy the data to copy onto the device
     * @param[in] spitch the stride length
     * @param[in] width the width of the 2D matrix to copy
     * @param[in] height the height of the 2D matrix to copy
     */
    virtual void copy_to_device_strided(const_host_pointer_type data_to_copy, std::size_t spitch, std::size_t width, std::size_t height) = 0;

    /**
     * @brief Copy device_ptr::size() many values from the device to the host buffer @p buffer.
     * @tparam layout the layout type of the matrix
     * @param[in] buffer the buffer to copy the data to
     * @throws plssvm::gpu_device_ptr_exception if @p buffer is too small to satisfy the copy
     */
    template <layout_type layout>
    void copy_to_host(matrix<value_type, layout> &buffer) const;

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

    /**
     * @brief Copy device_ptr::size() many values from the device to the device_ptr @p target (possibly located on another device).
     * @param[in] target the data to copy onto the device (possibly located on another device)
     */
    void copy_to_other_device(derived_gpu_device_ptr &target) const;
    /**
     * @brief Copy up-to @p count many values from the device to the device_ptr @p target (possibly located on another device) starting at device pointer position @p pos.
     * @details Copies `[pos, rcount)` values where `rcount` is the smaller value of @p count and `device_ptr::size() - pos`.
     * @param target the data to copy onto the device (possibly located on another device)
     * @param pos the starting position for the copying in the device pointer
     * @param count the number of elements to copy
     */
    virtual void copy_to_other_device(derived_gpu_device_ptr &target, size_type pos, size_type count) const = 0;

  protected:
    /// The device queue used to manage the device memory associated with this device pointer.
    queue_type queue_{};
    /// The size of the managed memory.
    plssvm::shape shape_{};
    /// The padding size of the managed memory.
    plssvm::shape padding_{};
    /// The device pointer pointing to the managed memory.
    device_pointer_type data_{};
    /// If true, use USM allocations automatically migrating the data between host and device.
    bool use_usm_allocations_{};
};

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::gpu_device_ptr(const size_type size, const queue_type queue, const bool use_usm_allocations) :
    queue_{ queue },
    shape_{ plssvm::shape{ size, 1 } },
    use_usm_allocations_{ use_usm_allocations } { }

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::gpu_device_ptr(const plssvm::shape shape, const queue_type queue, const bool use_usm_allocations) :
    queue_{ queue },
    shape_{ shape },
    use_usm_allocations_{ use_usm_allocations } { }

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::gpu_device_ptr(const plssvm::shape shape, const plssvm::shape padding, const queue_type queue, const bool use_usm_allocations) :
    queue_{ queue },
    shape_{ shape },
    padding_{ padding },
    use_usm_allocations_{ use_usm_allocations } { }

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::gpu_device_ptr(gpu_device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, queue_type{}) },
    shape_{ std::exchange(other.shape_, plssvm::shape{}) },
    padding_{ std::exchange(other.padding_, plssvm::shape{}) },
    data_{ std::exchange(other.data_, device_pointer_type{}) },
    use_usm_allocations_{ std::exchange(other.use_usm_allocations_, false) } { }

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
auto gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::operator=(gpu_device_ptr &&other) noexcept -> gpu_device_ptr & {
    // guard against self-assignment
    if (this != std::addressof(other)) {
        queue_ = std::exchange(other.queue_, queue_type{});
        shape_ = std::exchange(other.shape_, plssvm::shape{});
        padding_ = std::exchange(other.padding_, plssvm::shape{});
        data_ = std::exchange(other.data_, device_pointer_type{});
        use_usm_allocations_ = std::exchange(other.use_usm_allocations_, false);
    }
    return *this;
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::swap(gpu_device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(shape_, other.shape_);
    std::swap(padding_, other.padding_);
    std::swap(data_, other.data_);
    std::swap(use_usm_allocations_, other.use_usm_allocations_);
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::memset(const int pattern, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->memset(pattern, pos, this->size_padded() * sizeof(value_type));
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::fill(const value_type value, const size_type pos) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->fill(value, pos, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
template <layout_type layout>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device(const matrix<value_type, layout> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (data_to_copy.size_padded() < this->size_padded()) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", this->size_padded(), data_to_copy.size_padded()) };
    }
    this->copy_to_device(data_to_copy.data());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_device(data_to_copy, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (data_to_copy.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->copy_to_device(data_to_copy.data(), pos, rcount);
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device(const_host_pointer_type data_to_copy) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_device(data_to_copy, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
template <layout_type layout>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device_strided(const matrix<value_type, layout> &data_to_copy, const std::size_t start_row, const std::size_t num_rows) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (start_row + num_rows > data_to_copy.num_rows()) {
        throw gpu_device_ptr_exception{ fmt::format("Tried to copy lines {}-{} (zero-based index) to the device, but the matrix has only {} lines!", start_row, start_row + num_rows - 1, data_to_copy.num_rows()) };
    }
    if (num_rows * data_to_copy.num_cols() < this->size()) {
        throw gpu_device_ptr_exception{ fmt::format("Too few data to perform copy (needed: {}, provided: {})!", this->size(), num_rows * data_to_copy.num_cols()) };
    }

    if constexpr (layout == layout_type::aos) {
        // data is laid out linearly in memory -> no strides necessary -> directly copy to device
        this->copy_to_device(data_to_copy.data() + start_row * data_to_copy.num_cols_padded(), 0, num_rows * data_to_copy.num_cols_padded());
    } else {
        // data NOT laid out linearly in memory -> strides necessary
        if (num_rows == data_to_copy.num_rows()) {
            // use potential shortcut in strided memory copy -> only applicable if a single device is used, i.e., copying padding entries does not result in wrong values
            this->copy_to_device_strided(data_to_copy.data() + start_row, data_to_copy.num_rows_padded(), num_rows + data_to_copy.padding().x, data_to_copy.num_cols_padded());
        } else {
            // otherwise, perform actual strided copy
            this->copy_to_device_strided(data_to_copy.data() + start_row, data_to_copy.num_rows_padded(), num_rows, data_to_copy.num_cols_padded());
        }
    }
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_device_strided(const std::vector<value_type> &data_to_copy, std::size_t spitch, std::size_t width, std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (width > spitch) {
        throw gpu_device_ptr_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }
    if (width * height > data_to_copy.size()) {
        throw gpu_device_ptr_exception{ fmt::format("The sub-matrix ({}x{}) to copy is to big ({})!", width, height, data_to_copy.size()) };
    }

    this->copy_to_device_strided(data_to_copy.data(), spitch, width, height);
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
template <layout_type layout>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_host(matrix<value_type, layout> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    if (buffer.size_padded() < this->size_padded()) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", this->size_padded(), buffer.size_padded()) };
    }
    this->copy_to_host(buffer.data());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    this->copy_to_host(buffer, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (buffer.size() < rcount) {
        throw gpu_device_ptr_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->copy_to_host(buffer.data(), pos, rcount);
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_host(host_pointer_type buffer) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    this->copy_to_host(buffer, 0, this->size_padded());
}

template <typename T, typename queue_t, typename device_pointer_t, typename derived_gpu_device_ptr>
void gpu_device_ptr<T, queue_t, device_pointer_t, derived_gpu_device_ptr>::copy_to_other_device(derived_gpu_device_ptr &target) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != nullptr, "Invalid target pointer! Maybe target has been default constructed?");

    this->copy_to_other_device(target, 0, this->size_padded());
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_DEVICE_PTR_HPP_
