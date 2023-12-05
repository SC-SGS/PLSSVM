/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a matrix class used to hiding the data linearization using AoS and SoA.
 */

#ifndef PLSSVM_DETAIL_MATRIX_HPP_
#define PLSSVM_DETAIL_MATRIX_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"                                // PLSSVM_ASSERT
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log
#include "plssvm/detail/utility.hpp"                               // plssvm::detail::always_false_v
#include "plssvm/exceptions/exceptions.hpp"                        // plssvm::matrix_exception
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity_level

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <algorithm>    // std::equal, std::all_of, std::fill_n
#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <cstring>      // std::memcpy, std::memset
#include <iosfwd>       // std::istream forward declaration
#include <ostream>      // std::ostream
#include <string_view>  // std::string_view
#include <type_traits>  // std::enable_if, std::is_convertible_v, std::is_arithmetic_v
#include <utility>      // std::swap
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Enum class for all available layout types.
 */
enum class layout_type {
    /** Array-of-Structs (AoS) */
    aos,
    /** Structs-of-Arrays (SoA) */
    soa
};

/**
 * @brief Output the @p layout to the given output-stream @p out.
 * @param[in, out] out the output-stream to write the layout type to
 * @param[in] layout the layout type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, layout_type layout);

/**
 * @brief Use the input-stream @p in to initialize the @p layout type.
 * @param[in,out] in input-stream to extract the layout type from
 * @param[in] layout the layout type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, layout_type &layout);

/**
 * @brief In contrast to operator>> return the full name of the provided @p layout type.
 * @param[in] layout the layout type
 * @return the full name of the layout type (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view layout_type_to_full_string(layout_type layout);

/**
 * @brief A matrix class encapsulating a 1D array automatically handling indexing with AoS and SoA.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 */
template <typename T, layout_type layout_>
class matrix {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type!");

  public:
    /// The value type of the entries in this matrix.
    using value_type = T;
    /// The size type used in this matrix.
    using size_type = std::size_t;
    /// The reference type of the entries in this matrix (based on the value type).
    using reference = value_type &;
    /// The const reference type of the entries in this matrix (based on the value type).
    using const_reference = const value_type &;
    /// The pointer type of the entries in this matrix (based on the value type).
    using pointer = value_type *;
    /// The const pointer type of the entries in this matrix (based on the value type).
    using const_pointer = const value_type *;

    /**
     * @brief Default construct an empty matrix, i.e., zero rows and columns.
     */
    matrix() = default;

    /**
     * @brief Create a matrix of size @p num_rows x @p num_cols and default-initializes all values.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     */
    matrix(size_type num_rows, size_type num_cols);
    /**
     * @brief Create a matrix of size (@p num_rows + @p row_padding) x (@p num_cols + @p col_padding) and default-initializes all values.
     * @details The padding entries are always initialized to `0`!
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in] row_padding the number of padding values for each row
     * @param[in] col_padding the number of padding values for each column
     * @throws plssvm::matrix_exception if at least one of @p num_rows or @p num_cols is zero
     */
    matrix(size_type num_rows, size_type num_cols, size_type row_padding, size_type col_padding);

    /**
     * @brief Create a matrix of size @p num_rows x @p num_cols and initialize all entries with the value @p init.
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in] init the value of all entries in the matrix
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
    matrix(size_type num_rows, size_type num_cols, const U &init);
    /**
     * @brief Create a matrix of size (@p num_rows + @p row_padding) x (@p num_cols + @p col_padding) and initialize all valid entries with the value @p init.
     * @details The padding entries are always initialized to `0`!
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in] init the value of all entries in the matrix
     * @param[in] row_padding the number of padding values for each row
     * @param[in] col_padding the number of padding values for each column
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
    matrix(size_type num_rows, size_type num_cols, const U &init, size_type row_padding, size_type col_padding);

    /**
     * @brief Create a matrix of size @p num_rows x @p num_cols and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of cols in the matrix
     * @param[in] data the data values
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     * @throws plssvm::matrix_exception if @p num_rows times @p num_cols is not equal to the number of values in @p data
     */
    matrix(size_type num_rows, size_type num_cols, const std::vector<value_type> &data);
    /**
     * @brief Create a matrix of size (@p num_rows + @p row_padding) x (@p num_cols + @p col_padding) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The vector @p data must **not** contain padding entries!
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of cols in the matrix
     * @param[in] data the data values
     * @param[in] row_padding the number of padding values for each row
     * @param[in] col_padding the number of padding values for each column
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     * @throws plssvm::matrix_exception if @p num_rows times @p num_cols is not equal to the number of values in @p data
     */
    matrix(size_type num_rows, size_type num_cols, const std::vector<value_type> &data, size_type row_padding, size_type col_padding);

    /**
     * @brief Create a matrix of size @p num_rows x @p num_cols and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of cols in the matrix
     * @param[in] data the pointer to the data values
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     */
    matrix(size_type num_rows, size_type num_cols, const_pointer data);
    /**
     * @brief Create a matrix of size (@p num_rows + @p row_padding) x (@p num_cols + @p col_padding) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The data pointed to by @p data must **not** contain padding entries!
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of cols in the matrix
     * @param[in] data the pointer to the data values
     * @param[in] row_padding the number of padding values for each row
     * @param[in] col_padding the number of padding values for each column
     * @throws plssvm::matrix_exception if exactly one of @p num_rows or @p num_cols is zero; creates an empty matrix if both are zero
     */
    matrix(size_type num_rows, size_type num_cols, const_pointer data, size_type row_padding, size_type col_padding);

    /**
     * @brief Construct a new matrix from @p other. Respects potential different layout types.
     * @tparam other_layout_ the layout_type of the other matrix
     * @param[in] other the other matrix
     */
    template <layout_type other_layout_>
    explicit matrix(const matrix<T, other_layout_> &other);
    /**
     * @brief Construct a new matrix from @p other with the new padding sizes @p row_padding and @p col_padding. Respects potential different layout types.
     * @tparam other_layout_ the layout_type of the other matrix
     * @param[in] other the other matrix
     */
    template <layout_type other_layout_>
    matrix(const matrix<T, other_layout_> &other, size_type row_padding, size_type col_padding);

    /**
     * @brief Create a matrix from the provided 2D vector @p data.
     * @param[in] data the data used to initialize this matrix
     * @throws plssvm::matrix_exception if the data vectors contain different number of values
     * @throws plssvm::matrix_exception if one vector in the data vector is empty
     */
    explicit matrix(const std::vector<std::vector<value_type>> &data);
    /**
     * @brief Create a matrix from the provided 2D vector @p data including padding.
     * @note The two dimensional data vector @p data must **not** contain padding entries!
     * @param[in] data the data used to initialize this matrix
     * @param[in] row_padding the number of padding values for each row
     * @param[in] col_padding the number of padding values for each column
     */
    matrix(const std::vector<std::vector<value_type>> &data, size_type row_padding, size_type col_padding);

    /**
     * @brief Returns the shape of the matrix, i.e., the number of rows and columns.
     * @detail It holds: `m.shape().first == m.num_rows()` and `m.shape().second == m.num_cols()`.
     * @note **Doesn't** contain the padding sizes!
     * @return the shape of the matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] std::array<size_type, 2> shape() const noexcept { return { num_rows_, num_cols_ }; }
    /**
     * @brief Return the number of rows in the matrix.
     * @note **Doesn't** contain the row padding size!
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows() const noexcept { return num_rows_; }
    /**
     * @brief Return the number of columns in the matrix.
     * @note **Doesn't** contain the column padding size!
     * @return the number of columns (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols() const noexcept { return num_cols_; }
    /**
     * @brief Return the number of entries in the matrix.
     * @details It holds: `num_entries() == num_rows() * num_cols()`.
     * @note **Doesn't** contain the padding sizes!
     * @return the number of entries (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_entries() const noexcept { return num_rows_ * num_cols_; }
    /**
     * @brief Check whether the matrix is currently empty, i.e., has zero rows and columns.
     * @details This may only happen for a default initialized matrix.
     * @return `true` if the matrix is empty, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept { return num_rows_ == 0 && num_cols_ == 0; }

    /**
     * @brief Return the padding sizes for the rows and columns.
     * @return the padding sizes (`[[nodiscard]]`)
     */
    [[nodiscard]] std::array<size_type, 2> padding() const noexcept { return { row_padding_, col_padding_ }; }
    /**
     * @brief Returns the shape of the matrix including padding, i.e., the number of rows + row padding and columns + column padding.
     * @detail It holds: `m.shape_padded().first == m.num_rows_padded()` and `m.shape_padded().second == m.num_cols_padded()`.
     * @return the shape of the matrix including padding (`[[nodiscard]]`)
     */
    [[nodiscard]] std::array<size_type, 2> shape_padded() const noexcept { return { num_rows_ + row_padding_, num_cols_ + col_padding_ }; }
    /**
     * @brief Return the number of rows in the matrix including padding.
     * @return the number of rows + row padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows_padded() const noexcept { return num_rows_ + row_padding_; }
    /**
     * @brief Return the number of columns in the matrix including padding.
     * @return the number of columns + column padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols_padded() const noexcept { return num_cols_ + col_padding_; }
    /**
     * @brief Return the number of entries in the matrix including padding.
     * @details It holds: `num_entries_padded() == num_rows_padded() * num_cols_padded()`.
     * @return the number of entries (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_entries_padded() const noexcept { return (num_rows_ + row_padding_) * (num_cols_ + col_padding_); }
    /**
     * @brief Checks whether this matrix contains any padding entries.
     * @return `true` if this matrix is padded, `false` otherwise (`[[nodsicard]]`)
     */
    [[nodiscard]] bool is_padded() const noexcept { return !(row_padding_ == 0 && col_padding_ == 0); }

    /**
     * @brief Restore the padding entries, i.e., explicitly set all padding entries to `0` again.
     */
    void restore_padding() noexcept;

    /**
     * @brief Return the layout type used in this matrix.
     * @details The layout type is either Array-of-Structs (AoS) or Struct-of-Arrays (SoA)
     * @return the layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] static constexpr layout_type layout() noexcept { return layout_; }

    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type operator()(size_type row, size_type col) const;
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return a const reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference operator()(size_type row, size_type col);
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws plssvm::matrix_exception if the provided @p row is equal or large than the number of rows in the matrix
     * @throws plssvm::matrix_exception if the provided @p col is equal or large than the number of columns in the matrix
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type at(size_type row, size_type col) const;
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws plssvm::matrix_exception if the provided @p row is equal or large than the number of rows in the matrix
     * @throws plssvm::matrix_exception if the provided @p col is equal or large than the number of columns in the matrix
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference at(size_type row, size_type col);

    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] pointer data() noexcept { return data_.data(); }
    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] const_pointer data() const noexcept { return data_.data(); }
    /**
     * @brief Return the data as a 2D vector.
     * @return the two-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::vector<value_type>> to_2D_vector() const;
    /**
     * @brief Return the data as 2D vector including padding entries.
     * @return the two-dimensional data including padding (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::vector<value_type>> to_2D_vector_padded() const;

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other matrix to swap the entries from
     */
    void swap(matrix &other) noexcept;

  private:
    /**
     * @brief Copy the data from @p source to @p dest row- or column-wise depending on the current layout type.
     * @param[out] dest the destination buffer
     * @param[in] dest_shape the sizes of the destination buffer
     * @param[in] source the source buffer
     * @param[in] source_shape the sizes of the source buffer
     */
    void opt_mismatched_padding_copy(pointer dest, const std::array<size_type, 2> dest_shape, const_pointer source, const std::array<size_type, 2> source_shape) {
        if constexpr (layout_ == layout_type::aos) {
            // copy row-wise
            #pragma omp parallel for
            for (size_type row = 0; row < num_rows_; ++row) {
                std::memcpy(dest + row * dest_shape[1], source + row * source_shape[1], num_cols_ * sizeof(value_type));
            }
        } else if constexpr (layout_ == layout_type::soa) {
            // copy column-wise
            #pragma omp parallel for
            for (size_type col = 0; col < num_cols_; ++col) {
                std::memcpy(dest + col * dest_shape[0], source + col * source_shape[0], num_rows_ * sizeof(value_type));
            }
        } else {
            static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
        }
    }


    /// The number of rows.
    size_type num_rows_{ 0 };
    /// The number of padding values for each row.
    size_type row_padding_{ 0 };
    /// The number of columns.
    size_type num_cols_{ 0 };
    /// The number of padding values for each column.
    size_type col_padding_{ 0 };
    /// The (linearized, either in AoS or SoA layout) data.
    std::vector<value_type> data_{};
};

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols) :
    matrix{ num_rows, num_cols, value_type{} } {}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const size_type row_padding, const size_type col_padding) :
    num_rows_{ num_rows }, row_padding_{ row_padding }, num_cols_{ num_cols }, col_padding_{ col_padding }, data_(this->num_entries_padded(), value_type{}) {
    if (num_rows_ == 0 && num_cols_ != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (num_rows_ != 0 && num_cols_ == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, layout_type layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const U &init) :
    num_rows_{ num_rows }, row_padding_{ 0 }, num_cols_{ num_cols }, col_padding_{ 0 }, data_(this->num_entries(), static_cast<value_type>(init)) {
    if (num_rows_ == 0 && num_cols_ != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (num_rows_ != 0 && num_cols_ == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, layout_type layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const U &init, const size_type row_padding, const size_type col_padding) :
    num_rows_{ num_rows }, row_padding_{ row_padding }, num_cols_{ num_cols }, col_padding_{ col_padding }, data_(this->num_entries_padded(), static_cast<value_type>(0.0)) {
    if (num_rows_ == 0 && num_cols_ != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (num_rows_ != 0 && num_cols_ == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }

    if constexpr (layout_ == layout_type::aos) {
        // fill rows with values, respecting padding entries
        #pragma omp parallel for
        for (size_type row = 0; row < num_rows_; ++row) {
            std::fill_n(this->data() + row * this->num_cols_padded(), num_cols_, static_cast<value_type>(init));
        }
    } else if constexpr (layout_ == layout_type::soa) {
        // fill columns with values, respecting padding entries
        #pragma omp parallel for
        for (size_type col = 0; col < num_cols_; ++col) {
            std::fill_n(this->data() + col * this->num_rows_padded(), num_rows_, static_cast<value_type>(init));
        }
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const std::vector<value_type> &data) :
    matrix{ num_rows, num_cols } {
    if (this->num_entries() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->num_entries(), data.size()) };
    }

    // memcpy data to matrix
    std::memcpy(this->data(), data.data(), this->num_entries() * sizeof(value_type));
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const std::vector<value_type> &data, const size_type row_padding, const size_type col_padding) :
    matrix{ num_rows, num_cols, row_padding, col_padding } {
    if (this->num_entries() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->num_entries(), data.size()) };
    }

    // memcpy data row- or column-wise depending on the layout type to the matrix
    this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data.data(), this->shape());
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const_pointer data) :
    matrix{ num_rows, num_cols } {
    if (this->num_entries() > 0) {
        // memcpy data to matrix
        std::memcpy(this->data(), data, this->num_entries() * sizeof(value_type));
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const size_type num_rows, const size_type num_cols, const_pointer data, const size_type row_padding, const size_type col_padding) :
    matrix{ num_rows, num_cols, row_padding, col_padding } {
    if (this->num_entries() > 0) {
        // memcpy data row- or column-wise depending on the layout type to the matrix
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data, this->shape());
    }
}

template <typename T, layout_type layout_>
template <layout_type other_layout_>
matrix<T, layout_>::matrix(const matrix<T, other_layout_> &other) :
    matrix{ other.num_rows(), other.num_cols(), other.padding()[0], other.padding()[1] } {
    if constexpr (layout_ == other_layout_) {
        // same layout -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->num_entries_padded() * sizeof(value_type));
    } else {
        // convert AoS -> SoA or SoA -> AoS
        #pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows_; ++row) {
            for (size_type col = 0; col < num_cols_; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, layout_type layout_>
template <layout_type other_layout_>
matrix<T, layout_>::matrix(const matrix<T, other_layout_> &other, size_type row_padding, size_type col_padding) :
    matrix{ other.num_rows(), other.num_cols(), row_padding, col_padding } {
    if (layout_ == other_layout_ && this->padding() == other.padding()) {
        // same layout and same padding -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->num_entries_padded() * sizeof(value_type));
    } else if (layout_ == other_layout_) {
        // same layout but different padding -> memcpy each row separately
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), other.data(), other.shape_padded());
    } else {
        // convert AoS -> SoA or SoA -> AoS or manual copy because of mismatching padding sizes
        #pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows_; ++row) {
            for (size_type col = 0; col < num_cols_; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data) :
    matrix{ data, size_type{ 0 }, size_type{ 0 } } {}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data, const size_type row_padding, const size_type col_padding) :
    row_padding_{ row_padding }, col_padding_{ col_padding } {
    if (data.empty()) {
        // the provided 2D vector was empty -> set to empty matrix
        num_rows_ = 0;
        num_cols_ = 0;
        data_ = std::vector<value_type>{};
    } else {
        if (!std::all_of(data.cbegin(), data.cend(), [&data](const std::vector<value_type> &row) { return row.size() == data.front().size(); })) {
            throw matrix_exception{ "Each row in the matrix must contain the same amount of columns!" };
        }
        if (data.front().empty()) {
            throw matrix_exception{ "The data to create the matrix must at least have one column!" };
        }

        // the provided 2D vector contains at least one element -> initialize matrix
        num_rows_ = data.size();
        num_cols_ = data.front().size();
        data_ = std::vector<value_type>(this->num_entries_padded(), value_type{});

        if constexpr (layout_ == layout_type::aos) {
            // in case of AoS layout speed up conversion by using a simple memcpy over each row
            #pragma omp parallel for
            for (size_type row = 0; row < num_rows_; ++row) {
                std::memcpy(this->data() + row * this->num_cols_padded(), data[row].data(), num_cols_ * sizeof(value_type));
            }
        } else {
            // explicitly iterate all elements otherwise
            #pragma omp parallel for collapse(2)
            for (size_type row = 0; row < num_rows_; ++row) {
                for (size_type col = 0; col < num_cols_; ++col) {
                    (*this)(row, col) = data[row][col];
                }
            }
        }
    }
}

template <typename T, layout_type layout_>
void matrix<T, layout_>::restore_padding() noexcept {
    if constexpr (layout_ == layout_type::aos) {
        // restore padding row-wise
        #pragma omp parallel for
        for (size_type row = 0; row < num_rows_; ++row) {
            std::memset(this->data() + (row + 1) * this->num_cols_padded() - col_padding_, 0, col_padding_ * sizeof(value_type));
        }
        std::memset(this->data() + num_rows_ * this->num_cols_padded(), 0, row_padding_ * this->num_cols_padded() * sizeof(value_type));
    } else if constexpr (layout_ == layout_type::soa) {
        // restore padding column-wise
        #pragma omp parallel for
        for (size_type col = 0; col < num_cols_; ++col) {
            std::memset(this->data() + (col + 1) * this->num_rows_padded() - row_padding_, 0, row_padding_ * sizeof(value_type));
        }
        std::memset(this->data() + num_cols_ * this->num_rows_padded(), 0, col_padding_ * this->num_rows_padded() * sizeof(value_type));
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator()(const size_type row, const size_type col) const -> value_type {
    PLSSVM_ASSERT(row < this->num_rows_padded(), fmt::format("The current row ({}) must be smaller than the number of padded rows ({})!", row, this->num_rows_padded()));
    PLSSVM_ASSERT(col < this->num_cols_padded(), fmt::format("The current column ({}) must be smaller than the number of padded columns ({})!", col, this->num_cols_padded()));
    if constexpr (layout_ == layout_type::aos) {
        return data_[row * this->num_cols_padded() + col];
    } else if constexpr (layout_ == layout_type::soa) {
        return data_[col * this->num_rows_padded() + row];
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}
template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator()(const size_type row, const size_type col) -> reference {
    PLSSVM_ASSERT(row < this->num_rows_padded(), fmt::format("The current row ({}) must be smaller than the number of padded rows ({})!", row, this->num_rows_padded()));
    PLSSVM_ASSERT(col < this->num_cols_padded(), fmt::format("The current column ({}) must be smaller than the number of padded columns ({})!", col, this->num_cols_padded()));
    if constexpr (layout_ == layout_type::aos) {
        return data_[row * this->num_cols_padded() + col];
    } else if constexpr (layout_ == layout_type::soa) {
        return data_[col * this->num_rows_padded() + row];
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}
template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) const -> value_type {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, num_rows_, row_padding_) };
    } else if (row >= this->num_rows()) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: attempting to access padding row {} (only {} real rows exist)!\n",
                    row,
                    num_rows_);
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, num_cols_, col_padding_) };
    } else if (col >= this->num_cols()) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: attempting to access padding column {} (only {} real columns exist)!\n",
                    col,
                    num_cols_);
    }

    return (*this)(row, col);
}
template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) -> reference {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, num_rows_, row_padding_) };
    } else if (row >= this->num_rows()) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: attempting to access padding row {} (only {} real rows exist)!\n",
                    row,
                    num_rows_);
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, num_cols_, col_padding_) };
    } else if (col >= this->num_cols()) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: attempting to access padding column {} (only {} real columns exist)!\n",
                    col,
                    num_cols_);
    }

    return (*this)(row, col);
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::to_2D_vector() const -> std::vector<std::vector<value_type>> {
    std::vector<std::vector<value_type>> ret(num_rows_, std::vector<value_type>(num_cols_));
    if constexpr (layout_ == layout_type::aos) {
        // in case of AoS layout speed up conversion by using a simple memcpy over each row
        #pragma omp parallel for
        for (size_type row = 0; row < num_rows_; ++row) {
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols_padded(), num_cols_ * sizeof(value_type));
        }
    } else {
        // explicitly iterate all elements otherwise
        #pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows_; ++row) {
            for (size_type col = 0; col < num_cols_; ++col) {
                ret[row][col] = (*this)(row, col);
            }
        }
    }
    return ret;
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::to_2D_vector_padded() const -> std::vector<std::vector<value_type>> {
    std::vector<std::vector<value_type>> ret(this->num_rows_padded(), std::vector<value_type>(this->num_cols_padded(), value_type{}));
    if constexpr (layout_ == layout_type::aos) {
        // in case of AoS layout speed up conversion by using a simple memcpy over each row
        #pragma omp parallel for
        for (size_type row = 0; row < this->num_rows_padded(); ++row) {
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols_padded(), this->num_cols_padded() * sizeof(value_type));
        }
    } else {
        // explicitly iterate all elements otherwise
        #pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows_; ++row) {
            for (size_type col = 0; col < num_cols_; ++col) {
                ret[row][col] = (*this)(row, col);
            }
        }
    }
    return ret;
}

template <typename T, layout_type layout_>
void matrix<T, layout_>::swap(matrix<T, layout_> &other) noexcept {
    using std::swap;
    swap(this->num_rows_, other.num_rows_);
    swap(this->row_padding_, other.row_padding_);
    swap(this->num_cols_, other.num_cols_);
    swap(this->col_padding_, other.col_padding_);
    swap(this->data_, other.data_);
}
/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @tparam T the type of the matrix
 * @tparam layout the layout type provided at compile time (AoS or SoA)
 * @param[in, out] lhs the first matrix
 * @param[in,out] rhs the second matrix
 */
template <typename T, layout_type layout>
inline void swap(matrix<T, layout> &lhs, matrix<T, layout> &rhs) {
    lhs.swap(rhs);
}

/**
 * @brief Compares @p lhs and @p rhs for equality.
 * @details Comparing matrices with the same elements but different shapes, will return `false`.
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
 */
template <typename T, layout_type layout_>
[[nodiscard]] inline bool operator==(const matrix<T, layout_> &lhs, const matrix<T, layout_> &rhs) noexcept {
    return lhs.shape() == rhs.shape() && lhs.padding() == rhs.padding() && std::equal(lhs.data(), lhs.data() + lhs.num_entries(), rhs.data());
}
/**
 * @brief Compares @p lhs and @p rhs for inequality.
 * @details Comparing matrices with the same elements but different shapes, will return `true`.
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
 */
template <typename T, layout_type layout_>
[[nodiscard]] inline bool operator!=(const matrix<T, layout_> &lhs, const matrix<T, layout_> &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Output the matrix entries in @p matr to the output-stream @p out.
 * @details **Doesn't** output the padding entries.
 * @tparam T the type of the matrix
 * @tparam layout the layout type provided at compile time (AoS or SoA)
 * @param[in,out] out the output-stream to print the matrix entries to
 * @param[in] matr the matrix to print
 * @return the output-stream
 */
template <typename T, layout_type layout>
inline std::ostream &operator<<(std::ostream &out, const matrix<T, layout> &matr) {
    using size_type = typename matrix<T, layout>::size_type;
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            out << fmt::format("{:.10e} ", matr(row, col));
        }
        if (row < matr.num_rows() - 1) {
            out << '\n';
        }
    }
    return out;
}

/**
 * @brief Typedef for a matrix in Array-of-Struct (AoS) layout.
 */
template <typename T>
using aos_matrix = matrix<T, layout_type::aos>;
/**
 * @brief Typedef for a matrix in Struct-of-Array (SoA) layout.
 */
template <typename T>
using soa_matrix = matrix<T, layout_type::soa>;

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::layout_type> : fmt::ostream_formatter {};
template <typename T, plssvm::layout_type layout>
struct fmt::formatter<plssvm::matrix<T, layout>> : fmt::ostream_formatter {};

#endif  // PLSSVM_DETAIL_MATRIX_HPP_
