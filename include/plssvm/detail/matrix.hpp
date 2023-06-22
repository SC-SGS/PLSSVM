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

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/layout.hpp"          // plssvm::detail::layout_type
#include "plssvm/detail/utility.hpp"         // plssvm::detail::always_false_v
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::matrix_exception

#include "fmt/core.h"  // fmt::format

#include <cstddef>  // std::size_t
#include <ostream>  // std::ostream
#include <utility>  // std::pair, std::make_pair, std::swap
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief A matrix class encapsulating a 1D array automatically handling indexing with AoS and SoA.
 * @tparam T the type of the matrix
 * @tparam layout_ the layout type provided at compile time (AoS or SoA)
 */
template <typename T, layout_type layout_>
class matrix_impl {
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
     * @brief Create a matrix of size @p num_rows x @p num_cols and initialize all entries with the value @p init.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @param[in] init the value of all entries in the matrix
     * @throws plssvm::matrix_exception if at least one of @p num_rows or @p num_cols is zero
     */
    matrix_impl(const size_type num_rows, const size_type num_cols, const_reference init) :
        num_rows_{ num_rows }, num_cols_{ num_cols }, data_(num_rows * num_cols, init) {
        if (num_rows_ == 0) {
            throw matrix_exception{ "The number of rows is zero!" };
        }
        if (num_cols_ == 0) {
            throw matrix_exception{ "The number of columns is zero!" };
        }
    }
    /**
     * @brief Create a matrix of size @p num_rows x @p num_cols.
     * @param[in] num_rows the number of rows in the matrix
     * @param[in] num_cols the number of columns in the matrix
     * @throws plssvm::matrix_exception if at least one of @p num_rows or @p num_cols is zero
     */
    matrix_impl(const size_type num_rows, const size_type num_cols) :
        matrix_impl{ num_rows, num_cols, value_type{} } {}

    /**
     * @brief Returns the shape of the matrix, i.e., the number of rows and columns.
     * @detail It holds: `m.shape().first == m.num_rows()` and `m.shape().second == m.num_cols()`.
     * @return the shape of the matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] std::pair<size_type, size_type> shape() const noexcept {
        return std::make_pair(num_rows_, num_cols_);
    }
    /**
     * @brief Return the number of rows in the matrix.
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows() const noexcept {
        return num_rows_;
    }
    /**
     * @brief Return the number of columns in the matrix.
     * @return the number of columns (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols() const noexcept {
        return num_cols_;
    }
    /**
     * @brief Return the layout type used in this matrix.
     * @details The layout type is either Array-of-Structs (AoS) or Struct-of-Arrays (SoA)
     * @return the layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] layout_type layout() const noexcept {
        return layout_;
    }

    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type operator()(const size_type row, const size_type col) const {
        PLSSVM_ASSERT(row < num_rows_, fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, num_rows_));
        PLSSVM_ASSERT(col < num_cols_, fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, num_cols_));
        if constexpr (layout_ == layout_type::aos) {
            return data_[row * num_cols_ + col];
        } else if constexpr (layout_ == layout_type::soa) {
            return data_[col * num_rows_ + row];
        } else {
            static_assert(always_false_v<T>, "Unrecognized layout_type!");
        }
    }
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @return a const reference to the value (`[[nodiscard]]`)
     */
    reference operator()(const size_type row, const size_type col) {
        PLSSVM_ASSERT(row < num_rows_, fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, num_rows_));
        PLSSVM_ASSERT(col < num_cols_, fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, num_cols_));
        if constexpr (layout_ == layout_type::aos) {
            return data_[row * num_cols_ + col];
        } else if constexpr (layout_ == layout_type::soa) {
            return data_[col * num_rows_ + row];
        } else {
            static_assert(always_false_v<T>, "Unrecognized layout_type!");
        }
    }
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws plssvm::matrix_exception if the provided @p row is equal or large than the number of rows in the matrix
     * @throws plssvm::matrix_exception if the provided @p col is equal or large than the number of columns in the matrix
     * @return the value (`[[nodiscard]]`)
     */
    [[nodiscard]] value_type at(const size_type row, const size_type col) const {
        if (row >= num_rows_) {
            throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, num_rows_) };
        }
        if (col >= num_cols_) {
            throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns ({})!", row, num_rows_) };
        }

        return this->operator()(row, col);
    }
    /**
     * @brief Returns the value at @p row and @p col as defined by the matrix's layout type.
     * @param[in] row the value's row
     * @param[in] col the value's column
     * @throws plssvm::matrix_exception if the provided @p row is equal or large than the number of rows in the matrix
     * @throws plssvm::matrix_exception if the provided @p col is equal or large than the number of columns in the matrix
     * @return the value (`[[nodiscard]]`)
     */
    reference at(const size_type row, const size_type col) {
        if (row >= num_rows_) {
            throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, num_rows_) };
        }
        if (col >= num_cols_) {
            throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns ({})!", row, num_rows_) };
        }

        return this->operator()(row, col);
    }

    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] pointer data() noexcept {
        return data_.data();
    }
    /**
     * @brief Return a pointer to the underlying one-dimensional data structure.
     * @return the one-dimensional data (`[[nodiscard]]`)
     */
    [[nodiscard]] const_pointer data() const noexcept {
        return data_.data();
    }

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other matrix to swap the entries from
     */
    void swap(matrix_impl &other) noexcept {
        using std::swap;
        swap(this->num_rows_, other.num_rows_);
        swap(this->num_cols_, other.num_cols_);
        swap(this->data_, other.data_);
    }

    /**
     * @brief Compares @p lhs and @p rhs for equality.
     * @details Comparing matrices with the same elements but different shapes, will return `false`.
     * @param[in] lhs the first matrix
     * @param[in] rhs the second matrix
     * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator==(const matrix_impl &lhs, const matrix_impl &rhs) noexcept {
        return lhs.num_rows_ == rhs.num_rows_ && lhs.num_cols_ == rhs.num_cols_ && lhs.data_ == rhs.data_;
    }
    /**
     * @brief Compares @p lhs and @p rhs for inequality.
     * @details Comparing matrices with the same elements but different shapes, will return `true`.
     * @param[in] lhs the first matrix
     * @param[in] rhs the second matrix
     * @return `true` if both matrices are equal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator!=(const matrix_impl &lhs, const matrix_impl &rhs) noexcept {
        return !(lhs == rhs);
    }

  private:
    /// The number of rows.
    size_type num_rows_{};
    /// The number of columns.
    size_type num_cols_{};
    /// The (linearized, either in AoS or SoA layout) data.
    std::vector<value_type> data_{};
};

/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @tparam T the type of the matrix
 * @tparam layout the layout type provided at compile time (AoS or SoA)
 * @param[in, out] lhs the first matrix
 * @param[in,out] rhs the second matrix
 */
template <typename T, layout_type layout>
void swap(matrix_impl<T, layout> &lhs, matrix_impl<T, layout> &rhs) {
    lhs.swap(rhs);
}

/**
 * @brief Output the matrix entries in @p matr to the output-stream @p out.
 * @tparam T the type of the matrix
 * @tparam layout the layout type provided at compile time (AoS or SoA)
 * @param[in,out] out the output-stream to print the matrix entries to
 * @param[in] matr the matrix to print
 * @return the output-stream
 */
template <typename T, layout_type layout>
std::ostream &operator<<(std::ostream &out, const matrix_impl<T, layout> &matr) {
    using size_type = typename matrix_impl<T, layout>::size_type;
    for (size_type row = 0; row < matr.num_rows(); ++row) {
        for (size_type col = 0; col < matr.num_cols(); ++col) {
            out << matr(row, col) << ' ';
        }
        out << '\n';
    }
    return out;
}

/**
 * @brief Typedef for a matrix in Array-of-Struct (AoS) layout.
 */
template <typename T>
using aos_matrix = matrix_impl<T, layout_type::aos>;
/**
 * @brief Typedef for a matrix in Struct-of-Array (SoA) layout.
 */
template <typename T>
using soa_matrix = matrix_impl<T, layout_type::soa>;
/**
 * @brief Typedef for a matrix in Array-of-Struct (AoS) layout.
 */
template <typename T>
using matrix = aos_matrix<T>;

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_MATRIX_HPP_
