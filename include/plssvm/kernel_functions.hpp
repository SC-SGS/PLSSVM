/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements the different kernel functions on the CPU using OpenMP.
 */

#ifndef PLSSVM_KERNEL_FUNCTIONS_HPP_
#define PLSSVM_KERNEL_FUNCTIONS_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // dot product, plssvm::operators::{squared_euclidean_dist, manhattan_dist}
#include "plssvm/detail/type_traits.hpp"     // plssvm::detail::always_false_v
#include "plssvm/detail/utility.hpp"         // plssvm::detail::get
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/gamma.hpp"                  // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::format

#include <cmath>    // std::pow, std::exp, std::fma, std::tanh
#include <cstddef>  // std::size_t
#include <variant>  // std::get
#include <vector>   // std::vector

namespace plssvm {

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
        return transposed<T>{ xi } * xj;
    } else if constexpr (kernel == kernel_function_type::polynomial) {
        // get parameters
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree_arg = static_cast<T>(detail::get<0>(args...));
        const auto gamma_arg = static_cast<T>(detail::get<1>(args...));
        const auto coef0_arg = static_cast<T>(detail::get<2>(args...));
        // perform kernel function calculation
        return std::pow(std::fma(gamma_arg, (transposed<T>{ xi } * xj), coef0_arg), degree_arg);
    } else if constexpr (kernel == kernel_function_type::rbf) {
        // get parameters
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        // perform kernel function calculation
        return std::exp(-gamma_arg * squared_euclidean_dist(xi, xj));
    } else if constexpr (kernel == kernel_function_type::sigmoid) {
        // get parameters
        static_assert(sizeof...(args) == 2, "Illegal number of additional parameters! Must be 2.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        const auto coef0_arg = static_cast<T>(detail::get<1>(args...));
        // perform kernel function calculation
        return std::tanh(std::fma(gamma_arg, transposed<T>{ xi } * xj, coef0_arg));
    } else if constexpr (kernel == kernel_function_type::laplacian) {
        // get parameters
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        // perform kernel function calculation
        return std::exp(-gamma_arg * manhattan_dist(xi, xj));
    } else if constexpr (kernel == kernel_function_type::chi_squared) {
        // get parameters
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        // perform kernel function calculation
        T res{ 0.0 };
        for (typename std::vector<T>::size_type i = 0; i < xi.size(); ++i) {
            const T sum = xi[i] + xj[i];
            if (sum != T{ 0.0 }) {
                const T temp = xi[i] - xj[i];
                res += (temp * temp) / sum;
            }
        }
        return std::exp(-gamma_arg * res);
    } else {
        static_assert(detail::always_false_v<Args...>, "Unknown kernel type!");
    }
}

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
[[nodiscard]] inline T kernel_function(const std::vector<T> &xi, const std::vector<T> &xj, const parameter &params) {
    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            return kernel_function<kernel_function_type::linear>(xi, xj);
        case kernel_function_type::polynomial:
            return kernel_function<kernel_function_type::polynomial>(xi, xj, params.degree, static_cast<T>(std::get<real_type>(params.gamma)), static_cast<T>(params.coef0));
        case kernel_function_type::rbf:
            return kernel_function<kernel_function_type::rbf>(xi, xj, static_cast<T>(std::get<real_type>(params.gamma)));
        case kernel_function_type::sigmoid:
            return kernel_function<kernel_function_type::sigmoid>(xi, xj, static_cast<T>(std::get<real_type>(params.gamma)), static_cast<T>(params.coef0));
        case kernel_function_type::laplacian:
            return kernel_function<kernel_function_type::laplacian>(xi, xj, static_cast<T>(std::get<real_type>(params.gamma)));
        case kernel_function_type::chi_squared:
            return kernel_function<kernel_function_type::chi_squared>(xi, xj, static_cast<T>(std::get<real_type>(params.gamma)));
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(params.kernel_type)) };
}

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
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp = std::fma(x(i, dim), y(j, dim), temp);
        }
        return temp;
    } else if constexpr (kernel == kernel_function_type::polynomial) {
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree_arg = static_cast<T>(detail::get<0>(args...));
        const auto gamma_arg = static_cast<T>(detail::get<1>(args...));
        const auto coef0_arg = static_cast<T>(detail::get<2>(args...));
        T temp{ 0.0 };
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp = std::fma(x(i, dim), y(j, dim), temp);
        }
        return std::pow(std::fma(gamma_arg, temp, coef0_arg), degree_arg);
    } else if constexpr (kernel == kernel_function_type::rbf) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        T temp{ 0.0 };
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            const T diff = x(i, dim) - y(j, dim);
            temp = std::fma(diff, diff, temp);
        }
        return std::exp(-gamma_arg * temp);
    } else if constexpr (kernel == kernel_function_type::sigmoid) {
        static_assert(sizeof...(args) == 2, "Illegal number of additional parameters! Must be 2.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        const auto coef0_arg = static_cast<T>(detail::get<1>(args...));
        T temp{ 0.0 };
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp = std::fma(x(i, dim), y(j, dim), temp);
        }
        return std::tanh(std::fma(gamma_arg, temp, coef0_arg));
    } else if constexpr (kernel == kernel_function_type::laplacian) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        T temp{ 0.0 };
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            temp += std::abs(x(i, dim) - y(j, dim));
        }
        return std::exp(-gamma_arg * temp);
    } else if constexpr (kernel == kernel_function_type::chi_squared) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma_arg = static_cast<T>(detail::get<0>(args...));
        T temp{ 0.0 };
        for (size_type dim = 0; dim < x.num_cols(); ++dim) {
            const T sum = x(i, dim) + y(j, dim);
            if (sum != T{ 0.0 }) {
                const T diff = x(i, dim) - y(j, dim);
                temp += (diff * diff) / sum;
            }
        }
        return std::exp(-gamma_arg * temp);
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
[[nodiscard]] inline T kernel_function(const matrix<T, layout> &x, const std::size_t i, const matrix<T, layout> &y, const std::size_t j, const parameter &params) {
    PLSSVM_ASSERT(x.num_cols() == y.num_cols(), "Sizes mismatch!: {} != {}", x.num_cols(), y.num_cols());
    PLSSVM_ASSERT(i < x.num_rows(), "Out-of-bounce access for i and x!: {} < {}", i, x.num_rows());
    PLSSVM_ASSERT(j < y.num_rows(), "Out-of-bounce access for j and y!: {} < {}", j, y.num_rows());

    switch (params.kernel_type) {
        case kernel_function_type::linear:
            return kernel_function<kernel_function_type::linear>(x, i, y, j);
        case kernel_function_type::polynomial:
            return kernel_function<kernel_function_type::polynomial>(x, i, y, j, params.degree, static_cast<T>(std::get<real_type>(params.gamma)), static_cast<T>(params.coef0));
        case kernel_function_type::rbf:
            return kernel_function<kernel_function_type::rbf>(x, i, y, j, static_cast<T>(std::get<real_type>(params.gamma)));
        case kernel_function_type::sigmoid:
            return kernel_function<kernel_function_type::sigmoid>(x, i, y, j, static_cast<double>(std::get<real_type>(params.gamma)), static_cast<double>(params.coef0));
        case kernel_function_type::laplacian:
            return kernel_function<kernel_function_type::laplacian>(x, i, y, j, static_cast<T>(std::get<real_type>(params.gamma)));
        case kernel_function_type::chi_squared:
            return kernel_function<kernel_function_type::chi_squared>(x, i, y, j, static_cast<T>(std::get<real_type>(params.gamma)));
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(params.kernel_type)) };
}

}  // namespace plssvm

#endif  // PLSSVM_KERNEL_FUNCTIONS_HPP_
