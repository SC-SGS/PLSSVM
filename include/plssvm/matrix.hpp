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
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log_untracked
#include "plssvm/detail/utility.hpp"                               // plssvm::detail::{always_false_v, unreachable}
#include "plssvm/exceptions/exceptions.hpp"                        // plssvm::matrix_exception
#include "plssvm/shape.hpp"                                        // plssvm::shape
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity_level

#include "fmt/base.h"     // fmt::formatter
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/format.h"   // fmt::format, fmt::runtime
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <algorithm>    // std::equal, std::all_of, std::fill_n
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
 * @brief A matrix class encapsulating a 1D array automatically handling indexing with AoS and SoA schemes.
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
     * @brief Create a matrix of size @p shape.x x @p shape.y and default-initializes all values.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    explicit matrix(plssvm::shape shape);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and default-initializes all values.
     * @details The padding entries are always initialized to `0`!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    matrix(plssvm::shape shape, plssvm::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize all entries with the value @p init.
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] init the value of all entries in the matrix
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    matrix(plssvm::shape shape, const U &init);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize all valid entries with the value @p init.
     * @details The padding entries are always initialized to `0`!
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] init the value of all entries in the matrix
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    matrix(plssvm::shape shape, const U &init, plssvm::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the data values
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero
     * @throws plssvm::matrix_exception if @p shape.x times @p shape.y is not equal to the number of values in @p data
     */
    matrix(plssvm::shape shape, const std::vector<value_type> &data);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The vector @p data must **not** contain padding entries!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the data values
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero
     * @throws plssvm::matrix_exception if @p shape.x times @p shape.y is not equal to the number of values in @p data
     */
    matrix(plssvm::shape shape, const std::vector<value_type> &data, plssvm::shape padding);

    /**
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the pointer to the data values
     * @throws plssvm::matrix_exception if exactly one of @p shape.x or @p shape.y is zero
     */
    matrix(plssvm::shape shape, const_pointer data);
    /**
     * @brief Create a matrix of size (@p shape.x + @p padding.x) x (@p shape.y + @p padding.y) and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @note The data pointed to by @p data must **not** contain padding entries!
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the pointer to the data values
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero
     */
    matrix(plssvm::shape shape, const_pointer data, plssvm::shape padding);

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
     * @param[in] padding the padding of the new matrix, i.e., the number of padding entries for each row and column
     */
    template <layout_type other_layout_>
    matrix(const matrix<T, other_layout_> &other, plssvm::shape padding);

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
     * @param[in] padding the padding of the matrix, i.e., the number of padding entries for each row and column
     */
    matrix(const std::vector<std::vector<value_type>> &data, plssvm::shape padding);

    /**
     * @brief Return the number of entries in the matrix **without** padding.
     * @details It holds: `size() == shape().x * shape().y`.
     * @return the number of entries **without** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept { return shape_.x * shape_.y; }

    /**
     * @brief Returns the shape of the matrix, i.e., the number of rows and columns **without** padding.
     * @return the shape of the matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape shape() const noexcept { return shape_; }

    /**
     * @brief Return the number of rows in the matrix **without** padding.
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows() const noexcept { return shape_.x; }

    /**
     * @brief Return the number of columns in the matrix **without** padding.
     * @return the number of columns (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols() const noexcept { return shape_.y; }

    /**
     * @brief Check whether the matrix is currently empty, i.e., has zero rows and columns.
     * @details This may only happen for a default initialized matrix or a matrix explicitly created with a shape of `{ 0, 0 }`.
     * @note A matrix with only padding entries is regarded as empty!
     * @return `true` if the matrix is empty, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept { return shape_.x == 0 && shape_.y == 0; }

    /**
     * @brief Return the padding sizes for the rows and columns.
     * @return the padding sizes (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape padding() const noexcept { return padding_; }

    /**
     * @brief Return the number of entries in the matrix **including** padding.
     * @details It holds: `size_padded() == shape_padded().x * shape_padded().y`.
     * @return the number of entries **including** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size_padded() const noexcept { return (shape_.x + padding_.x) * (shape_.y + padding_.y); }

    /**
     * @brief Returns the shape of the matrix **including** padding, i.e., the number of rows + row padding and columns + column padding.
     * @return the shape of the matrix **including** padding (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape shape_padded() const noexcept { return plssvm::shape{ shape_.x + padding_.x, shape_.y + padding_.y }; }

    /**
     * @brief Return the number of rows in the matrix **including** padding.
     * @return the number of rows + row padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows_padded() const noexcept { return shape_.x + padding_.x; }

    /**
     * @brief Return the number of columns in the matrix **including** padding.
     * @return the number of columns + column padding (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols_padded() const noexcept { return shape_.y + padding_.y; }

    /**
     * @brief Checks whether this matrix contains any padding entries.
     * @return `true` if this matrix is padded, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_padded() const noexcept { return !(padding_.x == 0 && padding_.y == 0); }

    /**
     * @brief Restore the padding entries, i.e., explicitly set all padding entries to `0` again.
     */
    void restore_padding() noexcept;

    /**
     * @brief Return the layout type used in this matrix.
     * @details The layout type is either Array-of-Structs (AoS) or Struct-of-Arrays (SoA).
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
     * @return a reference to the value (`[[nodiscard]]`)
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
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference at(size_type row, size_type col);

    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type operator[](size_type idx) const;
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference operator[](size_type idx);
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @throws plssvm::matrix_exception if the provided @p idx is equal or larger than the number of matrix entries
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type at(size_type idx) const;
    /**
     * @brief Returns the value at @p idx.
     * @param[in] idx the values index
     * @throws plssvm::matrix_exception if the provided @p idx is equal or larger than the number of matrix entries
     * @return a reference to the value (`[[nodiscard]]`)
     */
    [[nodiscard]] reference at(size_type idx);

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
     * @param[in] dest_shape the shape of the destination buffer
     * @param[in] source the source buffer
     * @param[in] source_shape the shape of the source buffer
     */
    void opt_mismatched_padding_copy(pointer dest, const plssvm::shape dest_shape, const_pointer source, const plssvm::shape source_shape) {
        if constexpr (layout_ == layout_type::aos) {
// copy row-wise
#pragma omp parallel for
            for (size_type row = 0; row < this->num_rows(); ++row) {
                std::memcpy(dest + row * dest_shape.y, source + row * source_shape.y, this->num_cols() * sizeof(value_type));
            }
        } else if constexpr (layout_ == layout_type::soa) {
// copy column-wise
#pragma omp parallel for
            for (size_type col = 0; col < this->num_cols(); ++col) {
                std::memcpy(dest + col * dest_shape.x, source + col * source_shape.x, this->num_rows() * sizeof(value_type));
            }
        } else {
            static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
        }
    }

    /// The shape of the matrix.
    plssvm::shape shape_{};
    /// The shape of the padding for each row and column.
    plssvm::shape padding_{};
    /// The (linearized, either in AoS or SoA layout) data.
    std::vector<value_type> data_{};
};

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape) :
    matrix{ shape, value_type{} } { }

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape, const plssvm::shape padding) :
    shape_{ shape },
    padding_{ padding },
    data_(this->size_padded(), value_type{}) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, layout_type layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const plssvm::shape shape, const U &init) :
    shape_{ shape },
    padding_{ 0, 0 },
    data_(this->size(), static_cast<value_type>(init)) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }
}

template <typename T, layout_type layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const plssvm::shape shape, const U &init, const plssvm::shape padding) :
    shape_{ shape },
    padding_{ padding },
    data_(this->size_padded(), static_cast<value_type>(0.0)) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
    }

    if constexpr (layout_ == layout_type::aos) {
// fill rows with values, respecting padding entries
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::fill_n(this->data() + row * this->num_cols_padded(), this->num_cols(), static_cast<value_type>(init));
        }
    } else if constexpr (layout_ == layout_type::soa) {
// fill columns with values, respecting padding entries
#pragma omp parallel for
        for (size_type col = 0; col < this->num_cols(); ++col) {
            std::fill_n(this->data() + col * this->num_rows_padded(), this->num_rows(), static_cast<value_type>(init));
        }
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape, const std::vector<value_type> &data) :
    matrix{ shape } {
    if (this->size() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->size(), data.size()) };
    }

    // memcpy data to matrix
    std::memcpy(this->data(), data.data(), this->size() * sizeof(value_type));
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape, const std::vector<value_type> &data, const plssvm::shape padding) :
    matrix{ shape, padding } {
    if (this->size() != data.size()) {
        throw matrix_exception{ fmt::format("The number of entries in the matrix ({}) must be equal to the size of the data ({})!", this->size(), data.size()) };
    }

    // memcpy data row- or column-wise depending on the layout type to the matrix
    this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data.data(), this->shape());
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape, const_pointer data) :
    matrix{ shape } {
    if (this->size() > 0) {
        // memcpy data to matrix
        std::memcpy(this->data(), data, this->size() * sizeof(value_type));
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape, const_pointer data, const plssvm::shape padding) :
    matrix{ shape, padding } {
    if (this->size() > 0) {
        // memcpy data row- or column-wise depending on the layout type to the matrix
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), data, this->shape());
    }
}

template <typename T, layout_type layout_>
template <layout_type other_layout_>
matrix<T, layout_>::matrix(const matrix<T, other_layout_> &other) :
    matrix{ other.shape(), other.padding() } {
    if constexpr (layout_ == other_layout_) {
        // same layout -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->size_padded() * sizeof(value_type));
    } else {
// convert AoS -> SoA or SoA -> AoS
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < this->num_rows(); ++row) {
            for (size_type col = 0; col < this->num_cols(); ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, layout_type layout_>
template <layout_type other_layout_>
matrix<T, layout_>::matrix(const matrix<value_type, other_layout_> &other, const plssvm::shape padding) :
    matrix{ other.shape(), padding } {
    if (layout_ == other_layout_ && this->padding() == other.padding()) {
        // same layout and same padding -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->size_padded() * sizeof(value_type));
    } else if (layout_ == other_layout_) {
        // same layout but different padding -> memcpy each row separately
        this->opt_mismatched_padding_copy(this->data(), this->shape_padded(), other.data(), other.shape_padded());
    } else {
// convert AoS -> SoA or SoA -> AoS or manual copy because of mismatching padding sizes
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < this->num_rows(); ++row) {
            for (size_type col = 0; col < this->num_cols(); ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data) :
    matrix{ data, plssvm::shape{ 0, 0 } } { }

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data, const plssvm::shape padding) :
    padding_{ padding } {
    if (data.empty()) {
        // the provided 2D vector was empty -> set to empty matrix
        shape_ = plssvm::shape{ 0, 0 };
        data_ = std::vector<value_type>(this->size_padded(), value_type{});
    } else {
        if (!std::all_of(data.cbegin(), data.cend(), [&data](const std::vector<value_type> &row) { return row.size() == data.front().size(); })) {
            throw matrix_exception{ "Each row in the matrix must contain the same amount of columns!" };
        }
        if (data.front().empty()) {
            throw matrix_exception{ "The data to create the matrix must at least have one column!" };
        }

        // the provided 2D vector contains at least one element -> initialize matrix
        shape_ = plssvm::shape{ data.size(), data.front().size() };
        data_ = std::vector<value_type>(this->size_padded(), value_type{});

        if constexpr (layout_ == layout_type::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
            for (size_type row = 0; row < this->num_rows(); ++row) {
                std::memcpy(this->data() + row * this->num_cols_padded(), data[row].data(), this->num_cols() * sizeof(value_type));
            }
        } else {
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
            for (size_type row = 0; row < this->num_rows(); ++row) {
                for (size_type col = 0; col < this->num_cols(); ++col) {
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
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::memset(this->data() + (row + 1) * this->num_cols_padded() - padding_.y, 0, padding_.y * sizeof(value_type));
        }
        std::memset(this->data() + this->num_rows() * this->num_cols_padded(), 0, padding_.x * this->num_cols_padded() * sizeof(value_type));
    } else if constexpr (layout_ == layout_type::soa) {
// restore padding column-wise
#pragma omp parallel for
        for (size_type col = 0; col < this->num_cols(); ++col) {
            std::memset(this->data() + (col + 1) * this->num_rows_padded() - padding_.x, 0, padding_.x * sizeof(value_type));
        }
        std::memset(this->data() + this->num_cols() * this->num_rows_padded(), 0, padding_.y * this->num_rows_padded() * sizeof(value_type));
    } else {
        static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
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
        static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
    }
    detail::unreachable();
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
    detail::unreachable();
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) const -> value_type {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, this->num_rows(), padding_.x) };
    } else if (row >= this->num_rows()) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              "WARNING: attempting to access padding row {} (only {} real rows exist)!\n",
                              row,
                              this->num_rows());
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, this->num_cols(), padding_.y) };
    } else if (col >= this->num_cols()) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              "WARNING: attempting to access padding column {} (only {} real columns exist)!\n",
                              col,
                              this->num_cols());
    }

    return (*this)(row, col);
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) -> reference {
    if (row >= this->num_rows_padded()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows including padding ({} + {})!", row, this->num_rows(), padding_.x) };
    } else if (row >= this->num_rows()) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              "WARNING: attempting to access padding row {} (only {} real rows exist)!\n",
                              row,
                              this->num_rows());
    }
    if (col >= this->num_cols_padded()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns including padding ({} + {})!", col, this->num_cols(), padding_.y) };
    } else if (col >= this->num_cols()) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              "WARNING: attempting to access padding column {} (only {} real columns exist)!\n",
                              col,
                              this->num_cols());
    }

    return (*this)(row, col);
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator[](const size_type idx) const -> value_type {
    PLSSVM_ASSERT(idx < this->size_padded(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()));
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator[](const size_type idx) -> reference {
    PLSSVM_ASSERT(idx < this->size_padded(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()));
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type idx) const -> value_type {
    if (idx >= this->size_padded()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()) };
    }
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type idx) -> reference {
    if (idx >= this->size_padded()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size_padded()) };
    }
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::to_2D_vector() const -> std::vector<std::vector<value_type>> {
    std::vector<std::vector<value_type>> ret(this->num_rows(), std::vector<value_type>(this->num_cols()));
    if constexpr (layout_ == layout_type::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
        for (size_type row = 0; row < this->num_rows(); ++row) {
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols_padded(), this->num_cols() * sizeof(value_type));
        }
    } else {
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < this->num_rows(); ++row) {
            for (size_type col = 0; col < this->num_cols(); ++col) {
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
        for (size_type row = 0; row < this->num_rows(); ++row) {
            for (size_type col = 0; col < this->num_cols(); ++col) {
                ret[row][col] = (*this)(row, col);
            }
        }
    }
    return ret;
}

template <typename T, layout_type layout_>
void matrix<T, layout_>::swap(matrix<value_type, layout_> &other) noexcept {
    using std::swap;
    swap(this->shape_, other.shape_);
    swap(this->padding_, other.padding_);
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
inline void swap(matrix<T, layout> &lhs, matrix<T, layout> &rhs) noexcept {
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
    return lhs.shape() == rhs.shape() && lhs.padding() == rhs.padding() && std::equal(lhs.data(), lhs.data() + lhs.size_padded(), rhs.data());
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
            out << fmt::format(fmt::runtime("{:.10e} "), matr(row, col));
        }
        if (row < matr.num_rows() - 1) {
            out << '\n';
        }
    }
    return out;
}

//*************************************************************************************************************************************//
//                                                      plssvm::matrix operations                                                      //
//*************************************************************************************************************************************//
/**
 * @brief Scale all elements in the matrix @p matr by @p scale.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in,out] matr the matrix to scale
 * @param[in] scale the scaling factor
 * @return reference to the scaled matrix
 */
template <typename T, layout_type layout>
matrix<T, layout> &operator*=(matrix<T, layout> &matr, const T scale) {
    using size_type = typename matrix<T, layout>::size_type;

#pragma omp parallel for collapse(2) default(none) shared(matr) firstprivate(scale)
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            matr(row, col) *= scale;
        }
    }
    return matr;
}

/**
 * @brief Return a new matrix equal to @p matr where all elements are scaled by @p scale.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] matr the value used for scaling
 * @param[in] scale the scaling factor
 * @return the scaled matrix (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator*(matrix<T, layout> matr, const T scale) {
    matr *= scale;
    return matr;
}

/**
 * @copydoc operator*(matrix<T, layout>, const T)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator*(const T scale, matrix<T, layout> matr) {
    return matr * scale;
}

/**
 * @brief Add the values of the matrix @p rhs to the matrix @p lhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the matrix to add the values of @p rhs to
 * @param[in] rhs the values to add to @p lhs
 * @return reference to @p lhs
 */
template <typename T, layout_type layout>
matrix<T, layout> &operator+=(matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ({} != {})", lhs.shape(), rhs.shape());
    using size_type = typename matrix<T, layout>::size_type;

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs)
    for (size_type row = 0; row < lhs.num_rows(); ++row) {
        for (size_type col = 0; col < lhs.num_cols(); ++col) {
            lhs(row, col) += rhs(row, col);
        }
    }
    return lhs;
}

/**
 * @brief Return a new matrix with the values being the sum of @p lhs and @p rhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return the matrix sum (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator+(matrix<T, layout> lhs, const matrix<T, layout> &rhs) {
    lhs += rhs;
    return lhs;
}

/**
 * @brief Subtract the values of the matrix @p rhs from the matrix @p lhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the matrix to subtract the values of @p rhs from
 * @param[in] rhs the values to subtract from @p lhs
 * @return reference to @p lhs
 */
template <typename T, layout_type layout>
matrix<T, layout> &operator-=(matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ({} != {})", lhs.shape(), rhs.shape());
    using size_type = typename matrix<T, layout>::size_type;

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs)
    for (size_type row = 0; row < lhs.num_rows(); ++row) {
        for (size_type col = 0; col < lhs.num_cols(); ++col) {
            lhs(row, col) -= rhs(row, col);
        }
    }
    return lhs;
}

/**
 * @brief Return a new matrix with the values being the difference of @p lhs and @p rhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return the matrix difference (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator-(matrix<T, layout> lhs, const matrix<T, layout> &rhs) {
    lhs -= rhs;
    return lhs;
}

/**
 * @brief Perform a matrix-matrix multiplication between @p lhs and @p rhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return the matrix product (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator*(const matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.num_cols() == rhs.num_rows(), "Error: shapes missmatch! ({} (num_cols) != {} (num_rows))", lhs.num_cols(), rhs.num_rows());
    using size_type = typename matrix<T, layout>::size_type;
    matrix<T, layout> res{ plssvm::shape{ lhs.num_rows(), rhs.num_cols() } };

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs, res)
    for (size_type row = 0; row < res.num_rows(); ++row) {
        for (size_type col = 0; col < res.num_cols(); ++col) {
            T temp{ 0.0 };
            for (size_type dim = 0; dim < lhs.num_cols(); ++dim) {
                temp = std::fma(lhs(row, dim), rhs(dim, col), temp);
            }
            res(row, col) = temp;
        }
    }
    return res;
}

/**
 * @brief Perform a rowwise dot product between @p lhs and @p rhs.
 * @details Essentially performs dot(@p lhs[i], @p rhs[i]).
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the first matrix
 * @param[in] rhs the second matrix
 * @return a vector containing the rowwise dot products (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] std::vector<T> rowwise_dot(const matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ({} != {})", lhs.shape(), rhs.shape());
    using size_type = typename matrix<T, layout>::size_type;
    std::vector<T> res(lhs.num_rows());

#pragma omp parallel for default(none) shared(res, lhs, rhs)
    for (size_type row = 0; row < res.size(); ++row) {
        T temp{ 0.0 };
        for (size_type col = 0; col < lhs.num_cols(); ++col) {
            temp = std::fma(lhs(row, col), rhs(row, col), temp);
        }
        res[row] = temp;
    }
    return res;
}

/**
 * @brief Return a new matrix that is the rowwise scale of the matrix @p matr with @p scale, i.e., row `i` of @p matr is scaled by @p scale[i].
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] scale the scaling values
 * @param[in] matr the matrix to scale
 * @return the newly scaled matrix (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> rowwise_scale(const std::vector<T> &scale, matrix<T, layout> matr) {
    PLSSVM_ASSERT(scale.size() == matr.num_rows(), "Error: shapes missmatch! ({} != {} (num_rows))", scale.size(), matr.num_rows());
    using size_type = typename matrix<T, layout>::size_type;

#pragma omp parallel for collapse(2) default(none) shared(matr, scale)
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            matr(row, col) *= scale[row];
        }
    }
    return matr;
}

/**
 * @brief Return a new matrix that is the rowwise scale of the matrix @p matr with @p scale, i.e., row `i` of @p matr is scaled by @p scale[i] if @p mask[i] is `true`.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] mask the mask
 * @param[in] scale the scaling values
 * @param[in] matr the matrix to scale
 * @return the newly scaled matrix (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> masked_rowwise_scale(const std::vector<unsigned long long> &mask, const std::vector<T> &scale, matrix<T, layout> matr) {
    PLSSVM_ASSERT(scale.size() == matr.num_rows(), "Error: shapes missmatch! ({} != {} (num_rows))", scale.size(), matr.num_rows());
    PLSSVM_ASSERT(mask.size() == matr.num_rows(), "Error: shapes missmatch! ({} != {} (num_rows))", mask.size(), matr.num_rows());
    using size_type = typename matrix<T, layout>::size_type;

    std::vector<T> masked_scale{ scale };
#pragma omp parallel for default(none) shared(mask, scale, masked_scale)
    for (size_type row = 0; row < scale.size(); ++row) {
        if (mask[row] == 0) {
            masked_scale[row] = T{ 0.0 };
        }
    }
    return rowwise_scale(masked_scale, std::move(matr));
}

/**
 * @brief Calculate the variance of the matrix @p matr.
 * @details Used formula: \f$var = \frac{1}{n}\sum\limits_{i = 1}^n (x_i - \mu)^2\f$
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] matr the matrix to calculate the variance for
 * @return the matrix's variance (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] T variance(const matrix<T, layout> &matr) {
    using size_type = typename matrix<T, layout>::size_type;

    // calculate the mean of the matrix
    T mean{};
#pragma omp parallel for collapse(2) reduction(+ : mean)
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            mean += matr(row, col);
        }
    }
    mean /= static_cast<T>(matr.size());

    // calculate the variance of the matrix using the previous calculated mean
    T var{};
#pragma omp parallel for collapse(2) reduction(+ : var)
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            const T diff = matr(row, col) - mean;
            var += diff * diff;
        }
    }
    return var / static_cast<T>(matr.size());
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

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::layout_type> : fmt::ostream_formatter { };

/**
 * @brief Custom {fmt} formatter for a plssvm::matrix. Doesn't print padding entries per default, but introduces the `p` format specifier which enables printing padding entries.
 * @tparam T the type of the matrix
 * @tparam layout the layout type provided at compile time (AoS or SoA)
 */
template <typename T, plssvm::layout_type layout>
struct fmt::formatter<plssvm::matrix<T, layout>> {
    /// Save whether padding entries should be printed or not.
    bool with_padding_{ false };

    /**
     * @brief Parse the format specifier. If `p` is provided, padding entries will be formatted.
     * @tparam ParseContext the {fmt} lib parser context
     * @param[in,out] ctx the format specifiers
     * @return the iterator pointing past the end of the parsed format specifier
     */
    template <typename ParseContext>
    constexpr auto parse(ParseContext &ctx) {
        auto it = ctx.begin(), end = ctx.end();
        if (it == end) {
            return it;
        }
        // check whether the format specifier p has been provided
        if (*it == 'p') {
            with_padding_ = true;
            ++it;
        }
        return it;
    }

    /**
     * @brief Format a plssvm::matrix with or without padding according to the provided format specifiers.
     * @tparam FormatContext the {fmt} lib format context
     * @param[in] matr the plssvm::matrix to format
     * @param[in,out] ctx the format specifiers
     * @return the output iterator to write to
     */
    template <typename FormatContext>
    auto format(const plssvm::matrix<T, layout> &matr, FormatContext &ctx) const {
        using size_type = typename plssvm::matrix<T, layout>::size_type;

        auto it = ctx.out();
        if (with_padding_) {
            // if requested, output padding entries
            // don't format padding entries with scientific notation if the value is 0 (to reduce clutter)!
            for (size_type row = 0; row < matr.num_rows_padded(); ++row) {
                for (size_type col = 0; col < matr.num_cols_padded(); ++col) {
                    if ((row >= matr.num_rows() || col >= matr.num_cols()) && matr(row, col) == T{ 0.0 }) {
                        it = format_to(it, "0 ");
                    } else {
                        it = format_to(it, "{:.10e} ", matr(row, col));
                    }
                }
                if (row < matr.num_rows_padded() - 1) {
                    it = format_to(it, "\n");
                }
            }
        } else {
            // output matrix without padding
            for (size_type row = 0; row < matr.num_rows(); ++row) {
                for (size_type col = 0; col < matr.num_cols(); ++col) {
                    it = format_to(it, "{:.10e} ", matr(row, col));
                }
                if (row < matr.num_rows() - 1) {
                    it = format_to(it, "\n");
                }
            }
        }
        return it;
    }
};

/// @endcond

#endif  // PLSSVM_DETAIL_MATRIX_HPP_
