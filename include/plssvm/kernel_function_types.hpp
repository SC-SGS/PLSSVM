/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible kernel function types.
 */

#ifndef PLSSVM_KERNEL_FUNCTION_TYPES_HPP_
#define PLSSVM_KERNEL_FUNCTION_TYPES_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // dot product, plssvm::squared_euclidean_dist
#include "plssvm/detail/type_traits.hpp"     // plssvm::detail::always_false_v
#include "plssvm/detail/utility.hpp"         // plssvm::detail::get
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <cmath>        // std::pow, std::exp, std::fma
#include <cstddef>      // std::size_t
#include <iosfwd>       // forward declare std::ostream and std::istream
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Enum class for all implemented kernel functions.
 */
enum class kernel_function_type {
    /** Linear kernel function: \f$\vec{u}^T \cdot \vec{v}\f$. */
    linear = 0,
    /** Polynomial kernel function: \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$. */
    polynomial = 1,
    /** Radial basis function: \f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$. */
    rbf = 2
};

/**
 * @brief Output the @p kernel type to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the kernel type to
 * @param[in] kernel the kernel type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, kernel_function_type kernel);

/**
 * @brief Return the mathematical representation of the kernel_type @p kernel.
 * @details Uses placeholders for the scalar values and vectors.
 * @param[in] kernel the kernel type
 * @return the mathematical representation of @p kernel (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view kernel_function_type_to_math_string(kernel_function_type kernel) noexcept;

/**
 * @brief Use the input-stream @p in to initialize the @p kernel type.
 * @details The extracted value is matched case-insensitive and can be the integer value of the kernel_type.
 * @param[in,out] in input-stream to extract the kernel type from
 * @param[in] kernel the kernel type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, kernel_function_type &kernel);

/**
 * @brief Computes the value of the two vectors @p xi and @p xj using the @p kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam T the used type to perform the calculations
 * @tparam Args additional parameters used in the respective kernel function
 * @param[in] xi the first vector
 * @param[in] xj the second vector
 * @param[in] args additional parameters
 * @return the value computed by the @p kernel function (`[[nodiscard]]`)
 */
template <kernel_function_type kernel, typename T, typename... Args>
[[nodiscard]] inline T kernel_function(const std::vector<T> &xi, const std::vector<T> &xj, Args... args) {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    if constexpr (kernel == kernel_function_type::linear) {
        // get parameters
        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters! Must be 0.");
        // perform kernel function calculation
        return transposed{ xi } * xj;
    } else if constexpr (kernel == kernel_function_type::polynomial) {
        // get parameters
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree = static_cast<T>(detail::get<0>(args...));
        const auto gamma = static_cast<T>(detail::get<1>(args...));
        const auto coef0 = static_cast<T>(detail::get<2>(args...));
        // perform kernel function calculation
        return std::pow(std::fma(gamma, transposed{ xi } * xj, coef0), degree);
    } else if constexpr (kernel == kernel_function_type::rbf) {
        // get parameters
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma = static_cast<T>(detail::get<0>(args...));
        // perform kernel function calculation
        return std::exp(-gamma * squared_euclidean_dist(xi, xj));
    } else {
        static_assert(detail::always_false_v<Args...>, "Unknown kernel type!");
    }
}

// forward declare parameter class
struct parameter;

/**
 * @brief Computes the value of the two vectors @p xi and @p xj using the kernel function and kernel parameter stored in @p params.
 * @tparam T the used type to perform the calculations
 * @param[in] xi the first vector
 * @param[in] xj the second vector
 * @param[in] params class encapsulating the kernel type and kernel parameters
 * @throws plssvm::unsupported_kernel_type_exception if the kernel function in @p params is not supported
 * @return the computed kernel function value (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] T kernel_function(const std::vector<T> &xi, const std::vector<T> &xj, const parameter &params);

/**
 * @brief Computes the value of the two matrix rows @p i in @p x and @p j in @p y using the @p kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam T the used type to perform the calculations
 * @tparam layout the layout type of the two matrices
 * @tparam Args additional parameters used in the respective kernel function
 * @param[in] x the first matrix
 * @param[in] i the row in the first matrix
 * @param[in] y the second matrix
 * @param[in] j the row in the second matrix
 * @param[in] args additional parameters
 * @return the value computed by the @p kernel function (`[[nodiscard]]`)
 */
template <kernel_function_type kernel, typename T, layout_type layout, typename... Args>
[[nodiscard]] T kernel_function(const matrix<T, layout> &x, const std::size_t i, const matrix<T, layout> &y, const std::size_t j, Args... args) {
    PLSSVM_ASSERT(x.num_cols() == y.num_cols(), "Sizes mismatch!: {} != {}", x.num_cols(), y.num_cols());
    PLSSVM_ASSERT(i < x.num_rows(), "Out-of-bounce access for i and x!: {} < {}", i, x.num_rows());
    PLSSVM_ASSERT(j < y.num_rows(), "Out-of-bounce access for j and y!: {} < {}", j, y.num_rows());

    using size_type = typename matrix<T, layout>::size_type;

    if constexpr (kernel == kernel_function_type::linear) {
        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters! Must be 0.");
        T temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp = std::fma(x(i, dim), y(j, dim), temp);
        }
        return temp;
    } else if constexpr (kernel == kernel_function_type::polynomial) {
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree = static_cast<T>(detail::get<0>(args...));
        const auto gamma = static_cast<T>(detail::get<1>(args...));
        const auto coef0 = static_cast<T>(detail::get<2>(args...));
        T temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp = std::fma(x(i, dim), y(j, dim), temp);
        }
        return std::pow(std::fma(gamma, temp, coef0), degree);
    } else if constexpr (kernel == kernel_function_type::rbf) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma = static_cast<T>(detail::get<0>(args...));
        T temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            const T diff = x(i, dim) - y(j, dim);
            temp = std::fma(diff, diff, temp);
        }
        return std::exp(-gamma * temp);
    } else {
        static_assert(detail::always_false_v<Args...>, "Unknown kernel type!");
    }
}

/**
 * @brief Computes the value of the two matrix rows @p i in @p x and @p j in @p y using the kernel function and kernel parameter stored in @p params.
 * @tparam T the used type to perform the calculations
 * @tparam layout the layout type of the two matrices
 * @param[in] x the first matrix
 * @param[in] i the row in the first matrix
 * @param[in] y the second matrix
 * @param[in] j the row in the second matrix
 * @param[in] params class encapsulating the kernel type and kernel parameters
 * @throws plssvm::unsupported_kernel_type_exception if the kernel function in @p params is not supported
 * @return the computed kernel function value (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] T kernel_function(const matrix<T, layout> &x, std::size_t i, const matrix<T, layout> &y, std::size_t j, const parameter &params);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::kernel_function_type> : fmt::ostream_formatter {};

#endif  // PLSSVM_KERNEL_FUNCTION_TYPES_HPP_