/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement the different kernel functions on the GPU using Kokkos.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_KERNEL_KERNEL_FUNCTIONS_HPP_
#define PLSSVM_BACKENDS_KOKKOS_KERNEL_KERNEL_FUNCTIONS_HPP_

#include "plssvm/backends/Kokkos/detail/standard_layout_tuple.hpp"  // plssvm::kokkos::detail::standard_layout_tuple
#include "plssvm/constants.hpp"                                     // plssvm::real_type
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::always_false_v
#include "plssvm/kernel_function_types.hpp"                         // plssvm::kernel_function_type

#include "Kokkos_MathematicalFunctions.hpp"  // Kokkos::pow, Kokkos::exp, Kokkos::tanh, Kokkos::abs

#include <type_traits>  // std::is_same_v

namespace plssvm::kokkos::detail {

//***************************************************//
//                 feature reductions                //
//***************************************************//

/**
 * @brief Compute the default feature reduction, i.e., a simple dot-product.
 * @tparam kernel_function the kernel function type
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <kernel_function_type kernel_function>
KOKKOS_INLINE_FUNCTION real_type feature_reduce(const real_type val1, const real_type val2) {
    return val1 * val2;
}

/**
 * @brief Compute the feature reduction for the radial basis function kernel function, i.e., the squared euclidean distance.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <>
KOKKOS_INLINE_FUNCTION real_type feature_reduce<kernel_function_type::rbf>(const real_type val1, const real_type val2) {
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
KOKKOS_INLINE_FUNCTION real_type feature_reduce<kernel_function_type::laplacian>(const real_type val1, const real_type val2) {
    return ::Kokkos::fabs(val1 - val2);
}

/**
 * @brief Compute the feature reduction for the chi-squared kernel function.
 * @note Be sure that the denominator isn't 0.0 which may be the case for padding values.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
template <>
KOKKOS_INLINE_FUNCTION real_type feature_reduce<kernel_function_type::chi_squared>(const real_type val1, const real_type val2) {
    const real_type d = val1 - val2;
    if constexpr (std::is_same_v<real_type, float>) {
        return (real_type{ 1.0 } / (val1 + val2 + FLT_MIN)) * d * d;  // TODO: std::numeric_limits::min
    } else {
        return (real_type{ 1.0 } / (val1 + val2 + DBL_MIN)) * d * d;  // TODO: std::numeric_limits::min
    }
}

//***************************************************//
//                  kernel functions                 //
//***************************************************//

/**
 * @brief Compute the @p kernel_function using @p value and the @p params.
 * @tparam kernel_function the kernel function type
 * @tparam Args the types of the potential kernel function parameters
 * @param[in] value the value to apply the kernel function to
 * @param[in] params the potential kernel function parameters
 * @return the result value (`[[nodiscard]]`)
 */
template <kernel_function_type kernel_function, typename... Args>
KOKKOS_INLINE_FUNCTION real_type apply_kernel_function(const real_type value, const detail::standard_layout_tuple<Args...> params) {
    if constexpr (kernel_function == kernel_function_type::linear) {
        return value;
    } else if constexpr (kernel_function == kernel_function_type::polynomial) {
        return ::Kokkos::pow(detail::get<1>(params) * value + detail::get<2>(params), detail::get<0>(params));
    } else if constexpr (kernel_function == kernel_function_type::rbf) {
        return ::Kokkos::exp(-detail::get<0>(params) * value);
    } else if constexpr (kernel_function == kernel_function_type::sigmoid) {
        return ::Kokkos::tanh(detail::get<0>(params) * value + detail::get<1>(params));
    } else if constexpr (kernel_function == kernel_function_type::laplacian) {
        return ::Kokkos::exp(-detail::get<0>(params) * value);
    } else if constexpr (kernel_function == kernel_function_type::chi_squared) {
        return ::Kokkos::exp(-detail::get<0>(params) * value);
    } else {
        static_assert(::plssvm::detail::always_false_v<Args...>, "Unsupported kernel function!");
    }
}

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_KERNEL_KERNEL_FUNCTIONS_HPP_
