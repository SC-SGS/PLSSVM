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

#include "plssvm/constants.hpp"                                    // plssvm::real_type
#include "plssvm/default_value.hpp"                                // plssvm::default_value, plssvm::is_default_value_v
#include "plssvm/detail/igor_utility.hpp"                          // plssvm::detail::{has_only_named_args_v, get_value_from_named_parameter}
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log
#include "plssvm/detail/type_traits.hpp"                           // PLSSVM_REQUIRES, plssvm::detail::{remove_cvref_t, always_false_v}
#include "plssvm/detail/utility.hpp"                               // plssvm::detail::unreachable
#include "plssvm/kernel_function_types.hpp"                        // plssvm::kernel_function_type, plssvm::kernel_function_type_to_math_string
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity_level, plssvm::verbosity

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter
#include "igor/igor.hpp"  // IGOR_MAKE_NAMED_ARGUMENT, igor::parser, igor::has_unnamed_arguments, igor::has_other_than

#include <iosfwd>       // forward declare std::ostream and std::istream
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v, std::is_convertible_v
#include <utility>      // std::forward

namespace plssvm {

/// @cond Doxygen_suppress
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
/// Create a named argument for the used solver type.
IGOR_MAKE_NAMED_ARGUMENT(solver);
/// Create a named argument for the classification type used for fitting a model.
IGOR_MAKE_NAMED_ARGUMENT(classification);
/// Create a named argument for the SYCL backend specific SYCL implementation type (DPC++ or AdaptiveCpp).
IGOR_MAKE_NAMED_ARGUMENT(sycl_implementation_type);
/// Create a named argument for the SYCL backend specific kernel invocation type.
IGOR_MAKE_NAMED_ARGUMENT(sycl_kernel_invocation_type);
/// @endcond

namespace detail {

/**
 * @brief Trait to check whether @p Args only contains named-parameter that can be used to initialize a `plssvm::parameter` struct.
 */
template <typename... Args>
constexpr bool has_only_parameter_named_args_v = !igor::has_other_than<Args...>(plssvm::kernel_type, plssvm::gamma, plssvm::degree, plssvm::coef0, plssvm::cost);

/**
 * @brief Trait to check whether @p Args only contains named-parameter that can be used to initialize a `plssvm::parameter` struct including SYCL specific named-parameters.
 */
template <typename... Args>
constexpr bool has_only_sycl_parameter_named_args_v = !igor::has_other_than<Args...>(plssvm::kernel_type, plssvm::gamma, plssvm::degree, plssvm::coef0, plssvm::cost, plssvm::sycl_implementation_type, plssvm::sycl_kernel_invocation_type);

}  // namespace detail

/**
 * @example parameter_examples.cpp
 * @brief A few examples regarding the plssvm::parameter class.
 */

/**
 * @brief Class for encapsulating all important C-SVM parameters.
 */
struct parameter {
    /**
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

    /**
     * @brief Construct a parameter by using the values in @p params and overwriting all values using the provided named-parameters.
     * @tparam Args the type of the named-parameters
     * @param[in] params the parameters used to overwrite the default values
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_named_args_v<Args...>)>
    constexpr explicit parameter(const parameter &params, Args &&...named_args) :
        parameter{ params } {
        this->set_named_arguments(std::forward<Args>(named_args)...);
    }

    /**
     * @brief Construct a parameter set by overwriting the SVM parameters' default values that are provided using named-parameters.
     * @tparam Args the type of the named-parameters
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_named_args_v<Args...>)>
    constexpr explicit parameter(Args &&...named_args) noexcept {
        this->set_named_arguments(std::forward<Args>(named_args)...);
    }

    /// The used kernel function: linear, polynomial, or radial basis functions (rbf).
    default_value<kernel_function_type> kernel_type{ default_init<kernel_function_type>{ kernel_function_type::rbf } };
    /// The degree parameter used in the polynomial kernel function.
    default_value<int> degree{ default_init<int>{ 3 } };
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    default_value<real_type> gamma{ default_init<real_type>{ 0.0 } };
    /// The coef0 parameter used in the polynomial kernel function.
    default_value<real_type> coef0{ default_init<real_type>{ 0.0 } };
    /// The cost parameter in the C-SVM.
    default_value<real_type> cost{ default_init<real_type>{ 1.0 } };

    /**
     * @brief Checks whether the current parameter set is **equivalent** to the one given by @p other.
     * @details Compares the member variables based on the `kernel`, i.e., for example for the rbf kernel both parameter sets
     *          must **only** have the same `gamma` and `cost` values **but** may differ in the values of `degree` or `coef0`.
     *          If all members should be compared regardless of the kernel type, one can use the operator== overload.
     * @param[in] other the other parameter set compared to this one
     * @return `true` if both parameter sets are equivalent, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr bool equivalent(const parameter &other) const noexcept {
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

  private:
    /**
     * @brief Overwrite the default values of this parameter object with the potential provided named-parameters @p named_args.
     * @tparam Args the type of the named-parameters
     * @param[in] named_args the potential named-parameters
     */
    template <typename... Args>
    void set_named_arguments(Args &&...named_args) {
        igor::parser parser{ std::forward<Args>(named_args)... };

        // compile time check: only named parameter are permitted
        static_assert(!parser.has_unnamed_arguments(), "Can only use named parameter!");
        // compile time check: each named parameter must only be passed once
        static_assert(!parser.has_duplicates(), "Can only use each named parameter once!");
        // compile time check: only some named parameters are allowed
        static_assert(!parser.has_other_than(plssvm::kernel_type, plssvm::gamma, plssvm::degree, plssvm::coef0, plssvm::cost, plssvm::sycl_implementation_type, plssvm::sycl_kernel_invocation_type),
                      "An illegal named parameter has been passed!");

        // shorthand function for emitting a warning if a provided parameter is not used by the current kernel function
        [[maybe_unused]] const auto print_warning = [](const std::string_view param_name, const kernel_function_type kernel) {
            // NOTE: can't use the log function due to circular dependencies
            detail::log(verbosity_level::full | verbosity_level::warning,
                        "WARNING: {} parameter provided, which is not used in the {} kernel ({})!\n",
                        param_name,
                        kernel,
                        kernel_function_type_to_math_string(kernel));
        };

        // compile time/runtime check: the values must have the correct types
        if constexpr (parser.has(plssvm::kernel_type)) {
            // get the value of the provided named parameter
            kernel_type = detail::get_value_from_named_parameter<typename decltype(kernel_type)::value_type>(parser, plssvm::kernel_type);
        }
        if constexpr (parser.has(plssvm::gamma)) {
            // get the value of the provided named parameter
            gamma = detail::get_value_from_named_parameter<typename decltype(gamma)::value_type>(parser, plssvm::gamma);
            // runtime check: the value may only be used with a specific kernel type
            if (kernel_type == kernel_function_type::linear) {
                print_warning("gamma", kernel_type);
            }
        }
        if constexpr (parser.has(plssvm::degree)) {
            // get the value of the provided named parameter
            degree = detail::get_value_from_named_parameter<typename decltype(degree)::value_type>(parser, plssvm::degree);
            // runtime check: the value may only be used with a specific kernel type
            if (kernel_type == kernel_function_type::linear || kernel_type == kernel_function_type::rbf) {
                print_warning("degree", kernel_type);
            }
        }
        if constexpr (parser.has(plssvm::coef0)) {
            // get the value of the provided named parameter
            coef0 = detail::get_value_from_named_parameter<typename decltype(coef0)::value_type>(parser, plssvm::coef0);
            // runtime check: the value may only be used with a specific kernel type
            if (kernel_type == kernel_function_type::linear || kernel_type == kernel_function_type::rbf) {
                print_warning("coef0", kernel_type);
            }
        }
        if constexpr (parser.has(plssvm::cost)) {
            // get the value of the provided named parameter
            cost = detail::get_value_from_named_parameter<typename decltype(cost)::value_type>(parser, plssvm::cost);
        }
    }
};

/**
 * @brief Compares the two parameter sets @p lhs and @p rhs for equality.
 * @details Two parameter sets are equal if and only if **all** SVM parameters are equal.
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are equal, `false` otherwise (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr bool operator==(const parameter &lhs, const parameter &rhs) noexcept {
    return lhs.kernel_type == rhs.kernel_type && lhs.degree == rhs.degree && lhs.gamma == rhs.gamma && lhs.coef0 == rhs.coef0 && lhs.cost == rhs.cost;
}
/**
 * @brief Compares the two parameter sets @p lhs and @p rhs for inequality.
 * @details Two parameter sets are unequal if **any** of the SVM parameters are unequal.
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are unequal, `false` otherwise (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr bool operator!=(const parameter &lhs, const parameter &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Checks whether the two parameter sets @p lhs and @p rhs are **equivalent**.
 * @details Compares the member variables based on the `kernel`, i.e., for example for the rbf kernel both parameter sets
 *          must **only** have the same `gamma` and `cost` values **but** may differ in the values of `degree` or `coef0`.
 *          If all members should be compared regardless of the kernel type, one can use the operator== overload.
 * @param[in] lhs the first parameter set
 * @param[in] rhs the second parameter set
 * @return `true` if both parameter sets are equivalent, `false` otherwise (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr bool equivalent(const parameter &lhs, const parameter &rhs) noexcept {
    return lhs.equivalent(rhs);
}

/**
 * @brief Output all parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameter set
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parameter &params);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::parameter> : fmt::ostream_formatter {};

#endif  // PLSSVM_PARAMETER_HPP_