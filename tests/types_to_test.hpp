/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All type combinations that should be tested for a data set and model (and the corresponding helper functions) including utility functions.
 */

#ifndef PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#define PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#pragma once

#include "plssvm/detail/type_list.hpp"       // plssvm::detail::{real_type_list, label_type_list, real_type_label_type_combination_list}
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::layout_type

#include "gtest/gtest.h"                     // ::testing::Types

namespace util {

namespace detail {

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

/// A type list of all supported real types usable in google tests.
using real_type_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::real_type_list>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) usable in google tests.
using label_type_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::label_type_list>;

/// The cartesian product of all real types and label types usable in google test
using real_type_label_type_combination_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::real_type_label_type_combination_list>;

/**
 * @brief Encapsulates a combination of a used `real_type` (`float` or `double`) and a specific non-type template value (e.g., `plssvm::kernel_function_type` or `plssvm::layout_type`).
 * @tparam T the used `real_type`
 * @tparam non_type a non-type template value
 */
template <typename T, auto non_type>
struct parameter_definition {
    using real_type = T;
    static constexpr decltype(auto) value = non_type;
};

/// A type list of all supported real_type and kernel_function_type combinations.
using real_type_kernel_function_gtest = ::testing::Types<
    parameter_definition<float, plssvm::kernel_function_type::linear>,
    parameter_definition<float, plssvm::kernel_function_type::polynomial>,
    parameter_definition<float, plssvm::kernel_function_type::rbf>,
    parameter_definition<double, plssvm::kernel_function_type::linear>,
    parameter_definition<double, plssvm::kernel_function_type::polynomial>,
    parameter_definition<double, plssvm::kernel_function_type::rbf>>;

/// A type list of all supported real_type and layout_type combinations.
using real_type_layout_type_gtest = ::testing::Types<
    parameter_definition<float, plssvm::layout_type::aos>,
    parameter_definition<float, plssvm::layout_type::soa>,
    parameter_definition<double, plssvm::layout_type::aos>,
    parameter_definition<double, plssvm::layout_type::soa>>;

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_