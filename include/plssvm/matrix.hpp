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

#include "plssvm/detail/assert.hpp"                 // PLSSVM_ASSERT
#include "plssvm/detail/logging/log_untracked.hpp"  // plssvm::detail::log_untracked
#include "plssvm/detail/utility.hpp"                // plssvm::detail::{always_false_v, unreachable}
#include "plssvm/exceptions/exceptions.hpp"         // plssvm::matrix_exception
#include "plssvm/shape.hpp"                         // plssvm::shape
#include "plssvm/verbosity_levels.hpp"              // plssvm::verbosity_level

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
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize all entries with the value @p init.
     * @tparam U the type of the @p init value; must be convertible to @p T
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] init the value of all entries in the matrix
     * @throws plssvm::matrix_exception if exactly one of the @p shape values is zero; creates an empty matrix if both are zero
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    matrix(plssvm::shape shape, const U &init);

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
     * @brief Create a matrix of size @p shape.x x @p shape.y and initialize it to the values provided via @p data.
     * @note The underlying layout of @p data must be the same as the matrix layout since a simple `std::memcpy` is used.
     * @param[in] shape the shape of the matrix, i.e., the number of rows and columns
     * @param[in] data the pointer to the data values
     * @throws plssvm::matrix_exception if exactly one of @p shape.x or @p shape.y is zero
     */
    matrix(plssvm::shape shape, const_pointer data);

    /**
     * @brief Construct a new matrix from @p other. Respects potential different layout types.
     * @tparam other_layout_ the layout_type of the other matrix
     * @param[in] other the other matrix
     */
    template <layout_type other_layout_>
    explicit matrix(const matrix<T, other_layout_> &other);

    /**
     * @brief Create a matrix from the provided 2D vector @p data.
     * @param[in] data the data used to initialize this matrix
     * @throws plssvm::matrix_exception if the data vectors contain different number of values
     * @throws plssvm::matrix_exception if one vector in the data vector is empty
     */
    explicit matrix(const std::vector<std::vector<value_type>> &data);

    /**
     * @brief Return the number of entries in the matrix.
     * @details It holds: `size() == shape().x * shape().y`.
     * @return the number of entries (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type size() const noexcept { return shape_.x * shape_.y; }

    /**
     * @brief Returns the shape of the matrix, i.e., the number of rows and columns.
     * @return the shape of the matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::shape shape() const noexcept { return shape_; }

    /**
     * @brief Return the number of rows in the matrix.
     * @return the number of rows (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_rows() const noexcept { return shape_.x; }

    /**
     * @brief Return the number of columns in the matrix.
     * @return the number of columns (`[[nodiscard]]`)
     */
    [[nodiscard]] size_type num_cols() const noexcept { return shape_.y; }

    /**
     * @brief Check whether the matrix is currently empty, i.e., has zero rows and columns.
     * @details This may only happen for a default initialized matrix or a matrix explicitly created with a shape of `{ 0, 0 }`.
     * @return `true` if the matrix is empty, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool empty() const noexcept { return shape_.x == 0 && shape_.y == 0; }

    /**
     * @brief Return the layout type used in this matrix.
     * @details The layout type is either Array-of-Structs (AoS) or Struct-of-Arrays (SoA).
     * @return the layout type (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr static layout_type layout() noexcept { return layout_; }

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
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other matrix to swap the entries from
     */
    void swap(matrix &other) noexcept;

  private:
    /// The shape of the matrix.
    plssvm::shape shape_{};
    /// The (linearized, either in AoS or SoA layout) data.
    std::vector<value_type> data_{};
};

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const plssvm::shape shape) :
    matrix{ shape, value_type{} } { }

template <typename T, layout_type layout_>
template <typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool>>
matrix<T, layout_>::matrix(const plssvm::shape shape, const U &init) :
    shape_{ shape },
    data_(this->size(), static_cast<value_type>(init)) {
    if (this->num_rows() == 0 && this->num_cols() != 0) {
        throw matrix_exception{ "The number of rows is zero but the number of columns is not!" };
    }
    if (this->num_rows() != 0 && this->num_cols() == 0) {
        throw matrix_exception{ "The number of columns is zero but the number of rows is not!" };
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
matrix<T, layout_>::matrix(const plssvm::shape shape, const_pointer data) :
    matrix{ shape } {
    if (data == nullptr && this->size() > 0) {
        throw matrix_exception{ "The provided data pointer may not be a nullptr if the matrix size is greater than 0!" };
    }
    if (this->size() > 0) {
        // memcpy data to matrix
        std::memcpy(this->data(), data, this->size() * sizeof(value_type));
    }
}

template <typename T, layout_type layout_>
template <layout_type other_layout_>
matrix<T, layout_>::matrix(const matrix<T, other_layout_> &other) :
    matrix{ other.shape() } {
    if constexpr (layout_ == other_layout_) {
        // same layout -> simply memcpy underlying array
        std::memcpy(this->data(), other.data(), this->size() * sizeof(value_type));
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// convert AoS -> SoA or SoA -> AoS
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                (*this)(row, col) = other(row, col);
            }
        }
    }
}

template <typename T, layout_type layout_>
matrix<T, layout_>::matrix(const std::vector<std::vector<value_type>> &data) {
    if (data.empty()) {
        // the provided 2D vector was empty -> set to empty matrix
        shape_ = plssvm::shape{ 0, 0 };
        data_ = std::vector<value_type>(this->size(), value_type{});
    } else {
        if (!std::all_of(data.cbegin(), data.cend(), [&data](const std::vector<value_type> &row) { return row.size() == data.front().size(); })) {
            throw matrix_exception{ "Each row in the matrix must contain the same amount of columns!" };
        }
        if (data.front().empty()) {
            throw matrix_exception{ "The data to create the matrix must at least have one column!" };
        }

        // the provided 2D vector contains at least one element -> initialize matrix
        shape_ = plssvm::shape{ data.size(), data.front().size() };
        data_ = std::vector<value_type>(this->size(), value_type{});

        if constexpr (layout_ == layout_type::aos) {
// in case of AoS layout speed up conversion by using a simple memcpy over each row
#pragma omp parallel for
            for (size_type row = 0; row < this->num_rows(); ++row) {
                std::memcpy(this->data() + row * this->num_cols(), data[row].data(), this->num_cols() * sizeof(value_type));
            }
        } else {
            const size_type num_rows = this->num_rows();
            const size_type num_cols = this->num_cols();
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
            for (size_type row = 0; row < num_rows; ++row) {
                for (size_type col = 0; col < num_cols; ++col) {
                    (*this)(row, col) = data[row][col];
                }
            }
        }
    }
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator()(const size_type row, const size_type col) const -> value_type {
    PLSSVM_ASSERT(row < this->num_rows(), fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, this->num_rows()));
    PLSSVM_ASSERT(col < this->num_cols(), fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, this->num_cols()));
    if constexpr (layout_ == layout_type::aos) {
        return data_[row * this->num_cols() + col];
    } else if constexpr (layout_ == layout_type::soa) {
        return data_[col * this->num_rows() + row];
    } else {
        static_assert(detail::always_false_v<value_type>, "Unrecognized layout_type!");
    }
    detail::unreachable();
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator()(const size_type row, const size_type col) -> reference {
    PLSSVM_ASSERT(row < this->num_rows(), fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, this->num_rows()));
    PLSSVM_ASSERT(col < this->num_cols(), fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, this->num_cols()));
    if constexpr (layout_ == layout_type::aos) {
        return data_[row * this->num_cols() + col];
    } else if constexpr (layout_ == layout_type::soa) {
        return data_[col * this->num_rows() + row];
    } else {
        static_assert(detail::always_false_v<T>, "Unrecognized layout_type!");
    }
    detail::unreachable();
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) const -> value_type {
    if (row >= this->num_rows()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, this->num_rows()) };
    }
    if (col >= this->num_cols()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, this->num_cols()) };
    }

    return (*this)(row, col);
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type row, const size_type col) -> reference {
    if (row >= this->num_rows()) {
        throw matrix_exception{ fmt::format("The current row ({}) must be smaller than the number of rows ({})!", row, this->num_rows()) };
    }
    if (col >= this->num_cols()) {
        throw matrix_exception{ fmt::format("The current column ({}) must be smaller than the number of columns ({})!", col, this->num_cols()) };
    }

    return (*this)(row, col);
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator[](const size_type idx) const -> value_type {
    PLSSVM_ASSERT(idx < this->size(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size()));
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::operator[](const size_type idx) -> reference {
    PLSSVM_ASSERT(idx < this->size(), fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size()));
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type idx) const -> value_type {
    if (idx >= this->size()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size()) };
    }
    return data_[idx];
}

template <typename T, layout_type layout_>
auto matrix<T, layout_>::at(const size_type idx) -> reference {
    if (idx >= this->size()) {
        throw matrix_exception{ fmt::format("The current index ({}) must be smaller than the total number of matrix entries ({})!", idx, this->size()) };
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
            std::memcpy(ret[row].data(), this->data() + row * this->num_cols(), this->num_cols() * sizeof(value_type));
        }
    } else {
        const size_type num_rows = this->num_rows();
        const size_type num_cols = this->num_cols();
// explicitly iterate all elements otherwise
#pragma omp parallel for collapse(2)
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
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
    return lhs.shape() == rhs.shape() && std::equal(lhs.data(), lhs.data() + lhs.size(), rhs.data());
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

    const size_type num_rows = matr.num_rows();
    const size_type num_cols = matr.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(matr) firstprivate(scale, num_rows, num_cols)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

    const size_type num_rows = lhs.num_rows();
    const size_type num_cols = lhs.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs) firstprivate(num_rows, num_cols)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

    const size_type num_rows = lhs.num_rows();
    const size_type num_cols = lhs.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs) firstprivate(num_rows, num_cols)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

    const size_type num_rows = res.num_rows();
    const size_type num_cols = res.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(lhs, rhs, res) firstprivate(num_rows, num_cols)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

    const size_type num_rows = matr.num_rows();
    const size_type num_cols = matr.num_cols();

#pragma omp parallel for collapse(2) default(none) shared(matr, scale) firstprivate(num_rows, num_cols)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

    const size_type num_rows = matr.num_rows();
    const size_type num_cols = matr.num_cols();

    // calculate the mean of the matrix
    T mean{};
#pragma omp parallel for collapse(2) reduction(+ : mean)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            mean += matr(row, col);
        }
    }
    mean /= static_cast<T>(matr.size());

    // calculate the variance of the matrix using the previous calculated mean
    T var{};
#pragma omp parallel for collapse(2) reduction(+ : var)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
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

template <typename T, plssvm::layout_type layout>
struct fmt::formatter<plssvm::matrix<T, layout>> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_DETAIL_MATRIX_HPP_
