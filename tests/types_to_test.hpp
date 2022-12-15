/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All type combinations that should be tested for a data set (and the corresponding helper functions) including utility functions.
 */

#ifndef PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#define PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#pragma once

#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "gtest/gtest.h"  // ::testing::Types

#include <string>  // std::string
#include <tuple>   // std::tuple

namespace util {

/**
 * @brief Encapsulates a combination of a used `real_type` (`float` or `double`) and `label_type` (an arithmetic type or `std::string`).
 * @tparam T the used `real_type`
 * @tparam U the used `label_type`
 */
template <typename T, typename U>
struct real_type_label_type_combination {
    using real_type = T;
    using label_type = U;
};

namespace detail {

// concatenate the types of two tuples in a new tuple
template <typename S, typename T>
struct concat_tuple_types;
/**
 * @brief Concatenate the types of the two tuples to a new tuple type.
 * @tparam FirstTupleTypes the first tuple
 * @tparam SecondTupleTypes the second tuple
 */
template <typename... FirstTupleTypes, typename... SecondTupleTypes>
struct concat_tuple_types<std::tuple<FirstTupleTypes...>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<FirstTupleTypes..., SecondTupleTypes...>;
};
/**
 * @brief Shorthand for the `typename concat_tuple_types<...>::type` type.
 */
template <typename... T>
using concat_tuple_types_t = typename concat_tuple_types<T...>::type;

// calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding real_type_label_type_combination types.
template <typename S, typename T>
struct cartesian_type_product;
/**
 * @brief Calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding real_type_label_type_combination types.
 * @tparam FirstTupleType the first type in the first tuple (used to iterate all tuple types recursively)
 * @tparam FirstTupleRemainingTypes  the remaining types in the first tuple
 * @tparam SecondTupleTypes all types in the second tuple
 */
template <typename FirstTupleType, typename... FirstTupleRemainingTypes, typename... SecondTupleTypes>
struct cartesian_type_product<std::tuple<FirstTupleType, FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>> {
    // the cartesian product of {FirstTupleType} and {SecondTupleTypes...} is a list of real_type_label_type_combination
    using FirstTupleType_cross_SecondTupleTypes = std::tuple<real_type_label_type_combination<FirstTupleType, SecondTupleTypes>...>;

    // the cartesian product of {FirstTupleRemainingTypes...} and {Ts...} (computed recursively)
    using FirstTupleRemainingTypes_cross_SecondTupleTypes = typename cartesian_type_product<std::tuple<FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>>::type;

    // concatenate both products
    using type = concat_tuple_types_t<FirstTupleType_cross_SecondTupleTypes, FirstTupleRemainingTypes_cross_SecondTupleTypes>;
};
// end the recursion if the first tuple does not have any types left
template <typename... SecondTupleTypes>
struct cartesian_type_product<std::tuple<>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<>;  // the cartesian product of {}x{...} is {}
};
/**
 * @brief Shorthand for the `typename cartesian_type_product<...>::type` type.
 */
template <typename... T>
using cartesian_type_product_t = typename cartesian_type_product<T...>::type;

// convert the types in a tuple to GoogleTests ::testing::Type
template <typename Ts>
struct tuple_to_gtest_types;
/**
 * @brief Convert the types in a tuple to GoogleTests ::testing::Types.
 * @details For example: converts `std::tuple<int, long, float>` to `::testing::Types<int, long, float>`.
 * @tparam T the types in the tuple
 */
template <typename... T>
struct tuple_to_gtest_types<std::tuple<T...>> {
    using type = ::testing::Types<T...>;
};
/**
 * @brief Shorthand for the `typename tuple_to_gtest_types<...>::type` type.
 */
template <typename T>
using tuple_to_gtest_types_t = typename tuple_to_gtest_types<T>::type;

}  // namespace detail

/// A type list of all supported real types as `std::tuple`.
using real_type_list = std::tuple<float, double>;
/// A type list of all supported real types usable in google tests.
using real_type_gtest = detail::tuple_to_gtest_types_t<real_type_list>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using label_type_list = std::tuple<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double, std::string>;
/// A type list of all supported label types (currently arithmetic types and `std::string`) usable in google tests.
using label_type_gtest = detail::tuple_to_gtest_types_t<label_type_list>;

/// The cartesian product of all real types and label types as `std::tuple`.
using real_type_label_type_combination_list = detail::cartesian_type_product_t<real_type_list, label_type_list>;
/// The cartesian product of all real types and label types usable in google test
using real_type_label_type_combination_gtest = detail::tuple_to_gtest_types_t<real_type_label_type_combination_list>;

/**
 * @brief Encapsulates a combination of a used `real_type` (`float` or `double`) and a `plssvm::kernel_function_type`.
 * @tparam T the used `real_type`
 * @tparam kernel the `plssvm::kernel_function_type`
 */
template <typename T, plssvm::kernel_function_type kernel>
struct parameter_definition {
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

/// A type list of all supported real_type and kernel_function_type combinations.
using real_type_kernel_function_gtest = ::testing::Types<
    parameter_definition<float, plssvm::kernel_function_type::linear>,
    parameter_definition<float, plssvm::kernel_function_type::polynomial>,
    parameter_definition<float, plssvm::kernel_function_type::rbf>,
    parameter_definition<double, plssvm::kernel_function_type::linear>,
    parameter_definition<double, plssvm::kernel_function_type::polynomial>,
    parameter_definition<double, plssvm::kernel_function_type::rbf>>;

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_