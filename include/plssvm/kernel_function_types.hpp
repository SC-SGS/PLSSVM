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

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>       // forward declare std::ostream and std::istream
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief Enum class for all implemented kernel functions.
 */
enum class kernel_function_type {
    /** Linear kernel function: \f$\vec{u}^T \cdot \vec{v}\f$ */
    linear = 0,
    /** Polynomial kernel function: \f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$ */
    polynomial = 1,
    /** Radial basis function: \f$\exp(-gamma \cdot |\vec{u} - \vec{v}|^2)\f$ */
    rbf = 2,
    /** Sigmoid kernel function: \f$\tanh(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)\f$ */
    sigmoid = 3,
    /** Laplacian kernel function: \f$\exp(-gamma \cdot |\vec{u} - \vec{v}|_1)\f$ */
    laplacian = 4,
    /** Chi-squared kernel function (only well-defined for values > 0): \f$\exp(-gamma \cdot \sum_i \frac{(x[i] - y[i])^2}{x[i] + y[i]})\f$ */
    chi_squared = 5
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

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::kernel_function_type> : fmt::ostream_formatter { };

#endif  // PLSSVM_KERNEL_FUNCTION_TYPES_HPP_
