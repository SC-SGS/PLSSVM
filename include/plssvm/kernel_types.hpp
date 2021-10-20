/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines all available kernel types.
 */

#pragma once

#include "plssvm/detail/assert.hpp"     // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"  // dot product, plssvm::squared_euclidean_dist
#include "plssvm/detail/utility.hpp"    // plssvm::detail::always_false_v

#include <cmath>   // std::pow, std::exp, std::fma
#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm {

/**
 * @brief Enum class for all implemented kernel functions.
 */
enum class kernel_type {
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
std::ostream &operator<<(std::ostream &out, kernel_type kernel);

/**
 * @brief Use the input-stream @p in to initialize the @p kernel type.
 * @details The extracted value is matched case-insensitive and can be the integer value of the kernel_type.
 * @param[in,out] in input-stream to extract the kernel type from
 * @param[in] kernel the kernel type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, kernel_type &kernel);

/**
 * @brief Computes the value of the two vectors @p xi and @p xj using the @p kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam real_type the type of the values
 * @tparam Args additional parameters used in the respective kernel function
 * @param[in] xi the first vector
 * @param[in] xj the second vector
 * @param[in] args additional parameters
 * @return the value computed by the @p kernel function (`[[nodiscard]]`)
 */
template <kernel_type kernel, typename real_type, typename... Args>
[[nodiscard]] inline real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj, Args &&...args) {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    if constexpr (kernel == kernel_type::linear) {
        static_assert(sizeof...(args) == 0, "Illegal number of additional parameters! Must be 0.");
        return transposed{ xi } * xj;
    } else if constexpr (kernel == kernel_type::polynomial) {
        static_assert(sizeof...(args) == 3, "Illegal number of additional parameters! Must be 3.");
        const auto degree = static_cast<real_type>(detail::get<0>(args...));
        const auto gamma = static_cast<real_type>(detail::get<1>(args...));
        const auto coef0 = static_cast<real_type>(detail::get<2>(args...));
        return std::pow(std::fma(gamma, (transposed<real_type>{ xi } * xj), coef0), degree);
    } else if constexpr (kernel == kernel_type::rbf) {
        static_assert(sizeof...(args) == 1, "Illegal number of additional parameters! Must be 1.");
        const auto gamma = static_cast<real_type>(detail::get<0>(args...));
        return std::exp(-gamma * squared_euclidean_dist(xi, xj));
    } else {
        static_assert(detail::always_false_v<real_type>, "Unknown kernel type!");
    }
}

}  // namespace plssvm
