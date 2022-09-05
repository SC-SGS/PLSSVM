/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements the parameter class encapsulating all important C-SVM parameters.
 */

#pragma once

#include "plssvm/default_value.hpp"  // plssvm::default_value
#include "plssvm/kernel_types.hpp"   // plssvm::kernel_type

#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT

#include <iosfwd>       // forward declare std::ostream
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm {

// create named arguments
IGOR_MAKE_NAMED_ARGUMENT(gamma);
IGOR_MAKE_NAMED_ARGUMENT(degree);
IGOR_MAKE_NAMED_ARGUMENT(coef0);
IGOR_MAKE_NAMED_ARGUMENT(cost);
IGOR_MAKE_NAMED_ARGUMENT(epsilon);
IGOR_MAKE_NAMED_ARGUMENT(max_iter);
IGOR_MAKE_NAMED_ARGUMENT(sycl_implementation_type);
IGOR_MAKE_NAMED_ARGUMENT(sycl_kernel_invocation_type);

/**
 * @brief Base class for encapsulating all important C-SVM parameters.
 * @tparam T the real_type used
 */
template <typename T>
struct parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;

    parameter() = default;
    parameter(const kernel_type kernel_p, const int degree_p, const real_type gamma_p, const real_type coef0_p, const real_type cost_p) {
        // TODO: necessary? -> default_value constructor!
        kernel = kernel_p;
        degree = degree_p;
        gamma = gamma_p;
        coef0 = coef0_p;
        cost = cost_p;
    }

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    default_value<kernel_type> kernel = default_init<kernel_type>{ kernel_type::linear };
    /// The degree parameter used in the polynomial kernel function.
    default_value<int> degree = default_init<int>{ 3 };
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    default_value<real_type> gamma = default_init<real_type>{ 0.0 };
    /// The coef0 parameter used in the polynomial kernel function.
    default_value<real_type> coef0 = default_init<real_type>{ 0.0 };
    /// The cost parameter in the C-SVM.
    default_value<real_type> cost = default_init<real_type>{ 1.0 };

    /**
     * @brief Convert a `plssvm::parameter<T>`to a `plssvm::parameter<U>` (i.e., conversion between float <-> double).
     * @tparam U the type to convert to
     * @return the `plssvm::parameter` values converted to @p U
     */
    template <typename U>
    explicit operator parameter<U>() {
        // convert between parameter<float> <-> parameter<double>
        return parameter<U>{ kernel, degree, static_cast<U>(gamma.value()), static_cast<U>(coef0.value()), static_cast<U>(cost.value()) };
    }
};

extern template struct parameter<float>;
extern template struct parameter<double>;

// comparison operations
template <typename T>
constexpr bool operator==(const parameter<T> &lhs, const parameter<T> &rhs) noexcept {
    return lhs.kernel == rhs.kernel && lhs.degree == rhs.degree && lhs.gamma == rhs.gamma && lhs.coef0 == rhs.coef0 && lhs.cost == rhs.cost;
}
template <typename T>
constexpr bool operator!=(const parameter<T> &lhs, const parameter<T> &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Output all parameters encapsulated by @p params to the given output-stream @p out.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameters
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params);

}  // namespace plssvm