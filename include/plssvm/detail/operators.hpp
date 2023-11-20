/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines (arithmetic) functions on scalars, [`std::vector`](https://en.cppreference.com/w/cpp/container/vector), and plssvm::matrix.
 */

#ifndef PLSSVM_DETAIL_OPERATORS_HPP_
#define PLSSVM_DETAIL_OPERATORS_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"         // plssvm::matrix

#include <cmath>   // std::fma
#include <vector>  // std::vector

//*************************************************************************************************************************************//
//                                                          scalar operations                                                          //
//*************************************************************************************************************************************//

/**
 * @brief Returns +1 if x is positive and -1 if x is negative or 0.
 * @param[in] x the value to calculate the sign for
 * @return +1 if @p x is positive and -1 if @p x is negative or 0 (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline constexpr T sign(const T x) {
    return x > T{ 0 } ? T{ +1 } : T{ -1 };
}

//*************************************************************************************************************************************//
//                                                        std::vector operations                                                       //
//*************************************************************************************************************************************//

/**
 * @def PLSSVM_GENERATE_ARITHMETIC_OPERATION
 * @brief Generate arithmetic element-wise operations using @p Op for [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) (and scalars).
 * @details If available, OpenMP SIMD (`#pragma omp simd`) is used to speedup the computations.
 *
 * Given the variables
 * @code
 * std::vector<T> vec1 = ...;
 * std::vector<T> vec2 = ...;
 * T scalar = ...;
 * @endcode
 * the following operations for @p Op are generated (e.g., Op is `+`):
 * @code
 * vec1 += vec2;   // operator+=(vector, vector)
 * vec1 + vec2;    // operator+(vector, vector)
 * vec1 += scalar; // operator+=(vector, scalar)
 * vec1 + scalar;  // operator+(vector, scalar)
 * scalar + vec1;  // operator+(scalar, vector)
 * @endcode
 * @param[in] Op the operator to generate
 */
// clang-format off
#define PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION(Op)                                               \
    template <typename T>                                                                             \
    inline std::vector<T> &operator Op##=(std::vector<T> &lhs, const std::vector<T> &rhs) {           \
        PLSSVM_ASSERT(lhs.size() == rhs.size(), "Sizes mismatch!: {} != {}", lhs.size(), rhs.size()); \
                                                                                                      \
        _Pragma("omp simd")                                                                           \
        for (typename std::vector<T>::size_type i = 0; i < lhs.size(); ++i) {                         \
            lhs[i] Op##= rhs[i];                                                                      \
        }                                                                                             \
        return lhs;                                                                                   \
    }                                                                                                 \
    template <typename T>                                                                             \
    [[nodiscard]] inline std::vector<T> operator Op(std::vector<T> lhs, const std::vector<T> &rhs) {  \
        return lhs Op##= rhs;                                                                         \
    }                                                                                                 \
    template <typename T>                                                                             \
    inline std::vector<T> &operator Op##=(std::vector<T> &lhs, const T rhs) {                         \
        _Pragma("omp simd")                                                                           \
        for (typename std::vector<T>::size_type i = 0; i < lhs.size(); ++i) {                         \
            lhs[i] Op##= rhs;                                                                         \
        }                                                                                             \
        return lhs;                                                                                   \
    }                                                                                                 \
    template <typename T>                                                                             \
    [[nodiscard]] inline std::vector<T> operator Op(std::vector<T> lhs, const T rhs) {                \
        return lhs Op##= rhs;                                                                         \
    }                                                                                                 \
    template <typename T>                                                                             \
    [[nodiscard]] inline std::vector<T> operator Op(const T lhs, std::vector<T> rhs) {                \
        _Pragma("omp simd")                                                                           \
        for (typename std::vector<T>::size_type i = 0; i < rhs.size(); ++i) {                         \
            rhs[i] = lhs Op rhs[i];                                                                   \
        }                                                                                             \
        return rhs;                                                                                   \
    }
// clang-format on

namespace plssvm::operators {

// define arithmetic operations +-*/ on std::vector
PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION(+)
PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION(-)
PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION(*)
PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION(/)

/**
 * @brief Wrapper struct for overloading the dot product operator.
 * @details Used to calculate the dot product \f$x^T \cdot y\f$ using
 * @code
 * std::vector<T> x = ...;
 * std::vector<T> y = ...;
 * T res = transposed{ x } * y;  // same as: dot(x, y);
 * @endcode
 * @tparam T the value type
 */
template <typename T>
struct transposed {
    /// The encapsulated vector.
    const std::vector<T> &vec;
};
/**
 * @brief Deduction guide for the plssvm::operators::transposed struct needed for C++17.
 */
template <typename T>
transposed(const std::vector<T> &) -> transposed<T>;

/**
 * @brief Calculate the dot product (\f$x^T \cdot y\f$) between both [`std::vector`](https://en.cppreference.com/w/cpp/container/vector).
 * @details Explicitly uses `std::fma` for better performance and accuracy.
 * @tparam T the value type
 * @param[in] lhs the first vector
 * @param[in] rhs the second vector
 * @return the dot product (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T operator*(const transposed<T> &lhs, const std::vector<T> &rhs) {
    PLSSVM_ASSERT(lhs.vec.size() == rhs.size(), "Sizes mismatch!: {} != {}", lhs.vec.size(), rhs.size());

    T val{};
    #pragma omp simd reduction(+ : val)
    for (typename std::vector<T>::size_type i = 0; i < lhs.vec.size(); ++i) {
        val = std::fma(lhs.vec[i], rhs[i], val);
    }
    return val;
}

/**
 * @copydoc operator*(const transposed<T>&, const std::vector<T>&)
 */
template <typename T>
[[nodiscard]] inline T dot(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    return transposed{ lhs } * rhs;
}

/**
 * @brief Accumulate all elements in the [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) @p vec.
 * @details Uses OpenMP SIMD reduction to speedup the calculation.
 * @tparam T the value type
 * @param[in] vec the elements to accumulate
 * @return the sum of all elements (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T sum(const std::vector<T> &vec) {
    T val{};
    #pragma omp simd reduction(+ : val)
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i) {
        val += vec[i];
    }
    return val;
}

/**
 * @brief Calculates the squared Euclidean distance of both vectors: \f$d^2(x, y) = (x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots + (x_n - y_n)^2\f$.
 * @details Explicitly uses `std::fma` for better performance and accuracy.
 * @tparam T the value type
 * @param[in] lhs the first vector
 * @param[in] rhs the second vector
 * @return the squared euclidean distance (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T squared_euclidean_dist(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    PLSSVM_ASSERT(lhs.size() == rhs.size(), "Sizes mismatch!: {} != {}", lhs.size(), rhs.size());

    T val{};
    #pragma omp simd reduction(+ : val)
    for (typename std::vector<T>::size_type i = 0; i < lhs.size(); ++i) {
        const T diff = lhs[i] - rhs[i];
        val = std::fma(diff, diff, val);
    }
    return val;
}

#undef PLSSVM_GENERATE_VECTOR_ARITHMETIC_OPERATION

//*************************************************************************************************************************************//
//                                                      plssvm::matrix operations                                                      //
//*************************************************************************************************************************************//
/**
 * @brief Scale all elements in the matrix @p matr by @p scale.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in,out] matr the matrix to scale
 * @param[in] scale the scaling factor
 * @return reference to the scale matrix
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
 * @param[in] scale the scaling factor
 * @param[in] matr the value used for scaling
 * @return the scaled matrix (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator*(const T scale, matrix<T, layout> matr) {
    matr *= scale;
    return matr;
}
/**
 * @copydoc operator*(const T, matrix<T, layout>)
 */
template <typename T, layout_type layout>
[[nodiscard]] matrix<T, layout> operator*(matrix<T, layout> matr, const T scale) {
    return scale * matr;
}

/**
 * @brief Add the values of @p rhs to @p lhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the matrix to add the values of @p rhs to
 * @param[in] rhs the values to add to @p lhs
 * @return reference to @p lhs
 */
template <typename T, layout_type layout>
matrix<T, layout> &operator+=(matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ([{}] != [{}])", fmt::join(lhs.shape(), ", "), fmt::join(rhs.shape(), ", "));
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
 * @brief Subtract the values of @p rhs from @p lhs.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] lhs the matrix to subtract the values of @p rhs from
 * @param[in] rhs the values to subtract from @p lhs
 * @return reference to @p lhs
 */
template <typename T, layout_type layout>
matrix<T, layout> &operator-=(matrix<T, layout> &lhs, const matrix<T, layout> &rhs) {
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ([{}] != [{}])", fmt::join(lhs.shape(), ", "), fmt::join(rhs.shape(), ", "));
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
    matrix<T, layout> res{ lhs.num_rows(), rhs.num_cols() };

    #pragma omp parallel for collapse(2) default(none) shared(lhs, rhs, res)
    for (size_type row = 0; row < res.num_rows(); ++row) {
        for (size_type col = 0; col < res.num_cols(); ++col) {
            T temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (size_type dim = 0; dim < lhs.num_cols(); ++dim) {
                temp += lhs(row, dim) * rhs(dim, col);
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
    PLSSVM_ASSERT(lhs.shape() == rhs.shape(), "Error: shapes missmatch! ([{}] != [{}])", fmt::join(lhs.shape(), ", "), fmt::join(rhs.shape(), ", "));
    using size_type = typename matrix<T, layout>::size_type;
    std::vector<T> res(lhs.num_rows());

    #pragma omp parallel for default(none) shared(res, lhs, rhs)
    for (size_type row = 0; row < res.size(); ++row) {
        T temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (size_type col = 0; col < lhs.num_cols(); ++col) {
            temp += lhs(row, col) * rhs(row, col);
        }
        res[row] = temp;
    }
    return res;
}

/**
 * @brief Return a new matrix that is the rowwise scale of the matrix @p matr, i.e., row `i` of @p matr is scaled by @p scale[i].
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] scale the scaling values
 * @param[in] matr the matrix to scale
 * @return the scaled matrix (`[[nodiscard]]`)
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

}  // namespace plssvm::operators

#endif  // PLSSVM_DETAIL_OPERATORS_HPP_