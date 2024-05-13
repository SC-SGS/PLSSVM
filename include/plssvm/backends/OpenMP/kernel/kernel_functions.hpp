/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement the different kernel functions for the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_KERNEL_FUNCTIONS_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_KERNEL_FUNCTIONS_HPP_
#pragma once

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include <cmath>   // std::abs, std::pow, std::exp, std::tanh
#include <limits>  // std::numeric_limits::min

namespace plssvm::openmp::detail {

//***************************************************//
//                 feature reductions                //
//***************************************************//

/**
 * @brief Compute the default feature reduction, i.e., a simple dot-product.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <kernel_function_type kernel_function>
[[nodiscard]] inline real_type feature_reduce(const real_type val1, const real_type val2) {
    return val1 * val2;
}

/**
 * @brief Compute the feature reduction for the radial basis function kernel function, i.e., the squared Euclidean distance.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type feature_reduce<kernel_function_type::rbf>(const real_type val1, const real_type val2) {
    const real_type d = val1 - val2;
    return d * d;
}

/**
 * @brief Compute the feature reduction for the laplacian kernel function, i.e., the Manhattan distance.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type feature_reduce<kernel_function_type::laplacian>(const real_type val1, const real_type val2) {
    return std::abs(val1 - val2);
}

/**
 * @brief Compute the feature reduction for the chi-squared kernel function.
 * @note Be sure that the denominator isn't 0.0 which may be the case for padding values.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type feature_reduce<kernel_function_type::chi_squared>(const real_type val1, const real_type val2) {
    const real_type d = val1 - val2;
    return (real_type{ 1.0 } / (val1 + val2 + std::numeric_limits<real_type>::min())) * d * d;
}

//***************************************************//
//                  kernel functions                 //
//***************************************************//

/**
 * @brief Unimplemented base-template for all kernel functions.
 * @return the result value (`[[nodiscard]]`)
 */
template <kernel_function_type, typename... Args>
[[nodiscard]] inline real_type apply_kernel_function(real_type, Args...);

/**
 * @brief Compute the linear kernel function using @p value.
 * @param[in] value the value to apply the linear kernel function to
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::linear>(const real_type value) {
    return value;
}

/**
 * @brief Compute the polynomial kernel function using @p value.
 * @param[in] value the value to apply the polynomial kernel function to
 * @param[in] degree the degree parameter of the polynomial kernel function
 * @param[in] gamma the gamma parameter of the polynomial kernel function
 * @param[in] coef0 the coef0 parameter of the polynomial kernel function
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::polynomial>(const real_type value, const int degree, const real_type gamma, const real_type coef0) {
    return std::pow(gamma * value + coef0, (real_type) degree);
}

/**
 * @brief Compute the radial basis function kernel function using @p value.
 * @param[in] value the value to apply the rbf kernel function to
 * @param[in] gamma the gamma parameter of the rbf kernel function
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::rbf>(const real_type value, const real_type gamma) {
    return std::exp(-gamma * value);
}

/**
 * @brief Compute the sigmoid kernel function using @p value.
 * @param[in] value the value to apply the sigmoid kernel function to
 * @param[in] gamma the gamma parameter of the kernel kernel function
 * @param[in] coef0 the coef0 parameter of the kernel kernel function
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::sigmoid>(const real_type value, const real_type gamma, const real_type coef0) {
    return std::tanh(gamma * value + coef0);
}

/**
 * @brief Compute the laplacian function kernel function using @p value.
 * @param[in] value the value to apply the laplacian kernel function to
 * @param[in] gamma the gamma parameter of the laplacian kernel function
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::laplacian>(const real_type value, const real_type gamma) {
    return std::exp(-gamma * value);
}

/**
 * @brief Compute the chi-squared function kernel function using @p value.
 * @param[in] value the value to apply the chi-squared kernel function to
 * @param[in] gamma the gamma parameter of the chi-squared kernel function
 * @return the result value (`[[nodiscard]]`)
 */
template <>
[[nodiscard]] inline real_type apply_kernel_function<kernel_function_type::chi_squared>(const real_type value, const real_type gamma) {
    return std::exp(-gamma * value);
}

}  // namespace plssvm::openmp::detail

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_KERNEL_FUNCTIONS_HPP_