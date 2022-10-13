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

#ifndef PLSSVM_PARAMETER_HPP_
#define PLSSVM_PARAMETER_HPP_
#pragma once

#include "plssvm/default_value.hpp"          // plssvm::default_value
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT

#include <iosfwd>       // forward declare std::ostream
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm {

// create named arguments
/// Create a named argument for the `kernel` SVM parameter.
IGOR_MAKE_NAMED_ARGUMENT(kernel_type);
/// Create a named argument for the `gamma` SVM parameter.
IGOR_MAKE_NAMED_ARGUMENT(gamma);
/// Create a named argument for the `degree` SVM parameter.
IGOR_MAKE_NAMED_ARGUMENT(degree);
/// Create a named argument for the `coef0` SVM parameter.
IGOR_MAKE_NAMED_ARGUMENT(coef0);
/// Create a named argument for the `cost` SVM parameter.
IGOR_MAKE_NAMED_ARGUMENT(cost);
/// Create a named argument for the termination criterion `epsilon` of the CG algorithm.
IGOR_MAKE_NAMED_ARGUMENT(epsilon);
/// Create a named argument for the maximum number of iterations `max_iter` performed in the CG algorithm.
IGOR_MAKE_NAMED_ARGUMENT(max_iter);
/// Create a named argument for the SYCL backend specific SYCL implementation type (DPC++ or hipSYCL).
IGOR_MAKE_NAMED_ARGUMENT(sycl_implementation_type);
/// Create a named argument for the SYCL backend specific kernel invocation type (nd_range or hierarchical).
IGOR_MAKE_NAMED_ARGUMENT(sycl_kernel_invocation_type);

namespace detail {

/**
 * @brief Base class for encapsulating all important C-SVM parameters.
 * @tparam T the used real_type, must either be `float` or `double`
 */
template <typename T>
struct parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;

    /**double
     * @brief Default construct a parameter set, i.e., each SVM parameter has its default value.
     */
    constexpr parameter() noexcept = default;
    /**
     * @brief Construct a parameter set by explicitly overwriting the SVM parameters' default values.
     * @param[in] kernel_p the kernel type: linear, polynomial, or radial-basis functions (rbf)
     * @param[in] degree_p the degree used in the polynomial kernel function
     * @param[in] gamma_p the gamma used in the polynomial and rbf kernel functions
     * @param[in] coef0_p the coef0 used in the polynomial kernel function
     * @param[in] cost_p the cost used in all kernel functions
     */
    constexpr parameter(const kernel_function_type kernel_p, const int degree_p, const real_type gamma_p, const real_type coef0_p, const real_type cost_p) noexcept {
        // not in member initializer list since we want to override the default value
        kernel_type = kernel_p;
        degree = degree_p;
        gamma = gamma_p;
        coef0 = coef0_p;
        cost = cost_p;
    }

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    default_value<kernel_function_type> kernel_type{ default_init<kernel_function_type>{ kernel_function_type::linear } };
    /// The degree parameter used in the polynomial kernel function.
    default_value<int> degree{ default_init<int>{ 3 } };
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    default_value<real_type> gamma{ default_init<real_type>{ 0.0 } };
    /// The coef0 parameter used in the polynomial kernel function.
    default_value<real_type> coef0{ default_init<real_type>{ 0.0 } };
    /// The cost parameter in the C-SVM.
    default_value<real_type> cost{ default_init<real_type>{ 1.0 } };

    /**
     * @brief Convert a `plssvm::parameter<T>`to a `plssvm::parameter<U>` (i.e., conversion between float <-> double).
     * @tparam U the type to convert to
     * @return the `plssvm::parameter` values converted to @p U
     */
    template <typename U>
    constexpr explicit operator parameter<U>() const {
        if constexpr (std::is_same_v<U, real_type>) {
            // no special conversions needed
            return parameter<real_type>{ *this };
        } else {
            // convert between parameter<float> <-> parameter<double>
            return parameter<U>{ kernel_type, degree, default_value<U>{ gamma }, default_value<U>{ coef0 }, default_value<U>{ cost } };
        }
    }

    /**
     * @brief Checks whether the current parameter set is **equivalent** to the one given by @p other.
     * @details Compares the member variables based on the `kernel`, i.e., for example for the rbf kernel both parameter sets
     *          must **only** have the same `gamma` and `cost` values **but** may differ in the values of `degree` or `coef0`.
     *          If all members should be compared regardless of the kernel type, one can use the operator== overload.
     * @param[in] other the other parameter set to compare this one with
     * @return `true` if both parameter sets are equivalent, `false` otherwise (`[[nodiscard]]`)
     */
    constexpr bool equivalent(const parameter &other) const noexcept {
        // equality check, but only the member variables that a necessary for the current kernel type are compared!
        // cannot be equal if both parameters have different kernel types
        if (kernel_type != other.kernel_type) {
            return false;
        }
        // check member variables based on kernel type
        switch (kernel_type.value()) {
            case kernel_function_type::linear:
                return cost == other.cost;
            case kernel_function_type::polynomial:
                return degree == other.degree && gamma == other.gamma && coef0 == other.coef0 && cost == other.cost;
            case kernel_function_type::rbf:
                return gamma == other.gamma && cost == other.cost;
        }
        return false;
    }
};

extern template struct parameter<float>;
extern template struct parameter<double>;

/**
 * @brief Compares the two parameter sets @p lhs and @p rhs for equality.
 * @details Two parameter sets are equal if and only if **all** SVM parameters are equal.
 * @tparam T the real_type
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are equal, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator==(const parameter<T> &lhs, const parameter<T> &rhs) noexcept {
    return lhs.kernel_type == rhs.kernel_type && lhs.degree == rhs.degree && lhs.gamma == rhs.gamma && lhs.coef0 == rhs.coef0 && lhs.cost == rhs.cost;
}
/**
 * @brief Compares the two parameter sets @p lhs and @p rhs for inequality.
 * @details Two parameter sets are unequal if **any** of the SVM parameters are unequal.
 * @tparam T the real_type
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are unequal, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator!=(const parameter<T> &lhs, const parameter<T> &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Checks whether the two parameter sets @p lhs and @p rhs ares **equivalent**.
 * @details Compares the member variables based on the `kernel`, i.e., for example for the rbf kernel both parameter sets
 *          must **only** have the same `gamma` and `cost` values **but** may differ in the values of `degree` or `coef0`.
 *          If all members should be compared regardless of the kernel type, one can use the operator== overload.
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are equivalent, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool equivalent(const parameter<T> &lhs, const parameter<T> &rhs) noexcept {
    return lhs.equivalent(rhs);
}

/**
 * @brief Output all parameters encapsulated by @p params to the given output-stream @p out.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameter set
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params);

}

using parameter = detail::parameter<double>;

}  // namespace plssvm

#endif  // PLSSVM_PARAMETER_HPP_