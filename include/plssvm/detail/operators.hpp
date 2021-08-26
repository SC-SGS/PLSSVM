/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines (arithmetic) functions on [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) (and scalars).
 */

#pragma once

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include <vector>  // std::vector

/**
 * @def PLSSVM_GENERATE_ARITHMETIC_OPERATION
 * @brief Generate arithmetic element-wise operations using @p Op for [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) (and scalars).
 * @details If available, OpenMP SIMD (`#pragma omp simd`) is used to speedup the computations.
 *
 * Given the variables
 * @code
 * std::vector<T> vec1 = \dots;
 * std::vector<T> vec2 = \dots;
 * T scalar = \dots;
 * @endcode
 * the following operations for @p Op are generated (e.g. Op is `+`):
 * @code
 * vec1 += vec2;   // operator+=(vector, vector)
 * vec1 + vec2;    // operator+(vector, vector)
 * vec1 += scalar; // operator+=(vector, scalar)
 * vec1 + scalar;  // operator+(vector, scalar)
 * scalar + vec1;  // operator+(scalar, vector)
 * @endcode
 * Also checks that both vectors have the same size.
 * @param[in] Op the operator to generate
 */
// clang-format off
#define PLSSVM_GENERATE_ARITHMETIC_OPERATION(Op)                                                      \
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
        return rhs Op##= lhs;                                                                         \
    }
// clang-format on

namespace plssvm::operators {

// define arithmetic operations +-*/ on std::vector
PLSSVM_GENERATE_ARITHMETIC_OPERATION(+)
PLSSVM_GENERATE_ARITHMETIC_OPERATION(-)
PLSSVM_GENERATE_ARITHMETIC_OPERATION(*)
PLSSVM_GENERATE_ARITHMETIC_OPERATION(/)

/**
 * @brief Wrapper struct for overloading the dot product operator.
 * @details Used to calculate the dot product \f$x^T \cdot y\f$ using
 * @code
 * std::vector<T> x = \dots;
 * std::vector<T> y = \dots;
 * T res = transposed{ x } * y;  // dot(x, y);
 * @endcode
 * @tparam T the value type
 */
template <typename T>
struct transposed {
    /// The encapsulated vector.
    const std::vector<T> &vec;
};
/**
 * @brief Deduction guide needed for C++17.
 */
template <typename T>
transposed(const std::vector<T> &) -> transposed<T>;

/**
 * @brief Calculate the dot product (\f$x^T \cdot y\f$) between both [`std::vector`](https://en.cppreference.com/w/cpp/container/vector).
 * @details Uses OpenMP SIMD reduction to speedup the calculation.
 * @tparam T the value type
 * @param[in] lhs the first vector
 * @param[in] rhs the second vector
 * @return the dot product
 */
template <typename T>
[[nodiscard]] inline T operator*(const transposed<T> &lhs, const std::vector<T> &rhs) {
    PLSSVM_ASSERT(lhs.vec.size() == rhs.size(), "Sizes mismatch!: {} != {}", lhs.vec.size(), rhs.size());

    T val{};
    #pragma omp simd reduction(+:val)
    for (typename std::vector<T>::size_type i = 0; i < lhs.vec.size(); ++i) {
        // val += lhs.vec[i] * rhs[i]; //TODO: enable auto fma
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
 * @return the sum of all elements
 */
template <typename T>
[[nodiscard]] inline T sum(const std::vector<T> &vec) {
    T val{};
    #pragma omp simd reduction(+:val)
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i) {
        val += vec[i];
    }
    return val;
}

/**
 * @brief Calculates the squared Euclidean distance of both vectors: \f$d^2(x, y) = (x_1 - y_1)^2 + (x_2 - y_2)^2 + \dots + (x_n - y_n)^2\f$.
 * @details Uses OpenMP SIMD reduction to speedup the calculation.
 * @tparam T the value type
 * @param[in] lhs the first vector
 * @param[in] rhs the second vector
 * @return the squared euclidean distance
 */
template <typename T>
[[nodiscard]] inline T squared_euclidean_dist(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    PLSSVM_ASSERT(lhs.size() == rhs.size(), "Sizes mismatch!: {} != {}", lhs.size(), rhs.size());

    T val{};
    #pragma omp simd reduction(+:val)
    for (typename std::vector<T>::size_type i = 0; i < lhs.size(); ++i) {
        val += (lhs[i] - rhs[i]) * (lhs[i] - rhs[i]);
    }
    return val;
}


/**
* @copydoc inline constexpr int sign(T x)
 */
template <typename T>
[[nodiscard]] inline constexpr int sign(T x, std::false_type is_signed) {
    return T(0) < x;
}

/**
* @copydoc inline constexpr int sign(T x)
 */
template <typename T>
[[nodiscard]] inline constexpr int sign(T x, std::true_type is_signed) {
    return (T(0) < x) - (x < T(0));
}

/**
 * @brief Returns +1 if x is positive and -1 if x is negative.
 *
 * @param x the number parameter to evaluate
 * @return constexpr int +1 if x is positive and -1 if x is negative.
 */
template <typename T>
[[nodiscard]] inline constexpr int sign(T x) {
    return sign(x, std::is_signed<T>());
}

#undef PLSSVM_GENERATE_ARITHMETIC_OPERATION

}  // namespace plssvm::operators
