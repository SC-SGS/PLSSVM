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

#include "plssvm/classification_types.hpp"   // plssvm::classification_type
#include "plssvm/detail/type_list.hpp"       // plssvm::detail::{real_type_list, label_type_list, real_type_label_type_combination_list}
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::layout_type

#include "gtest/gtest.h"  // ::testing::Types

namespace util {

// TODO: naming

/**
 * @brief Encapsulates a combination of a used type and a specific non-type template value (e.g., `plssvm::kernel_function_type` or `plssvm::layout_type`).
 * @tparam T the used type
 * @tparam non_type a non-type template value
 */
template <typename T, auto non_type>
struct parameter_definition {
    using type = T;
    static constexpr decltype(auto) value = non_type;
};

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

/**
 * @brief Remove the first element of @p arr and return a new std::array with the same contents of @p arr except the first element.
 * @tparam T the type of the std::array elements
 * @tparam I the size of the array
 * @param[in] arr the array to remove the first value from
 * @return a new `std::array<T, I - 1>` (`[[nodiscard]]`)
 **/
template <typename T, std::size_t I>
constexpr std::array<T, I - 1> array_without_first_element(const std::array<T, I> &arr) {
    static_assert(I > 0, "Cannot remove an value from an empty array!");

    // 1. convert the std::array to an std::tuple
    // 2. remove the first element from the std::tuple
    // 3. convert the std::tuple back to an std::array
    return std::apply([](auto... elems) constexpr { return std::array<T, sizeof...(elems)>{ elems... }; },
                      std::apply([](auto, auto... rest) constexpr { return std::make_tuple(rest...); }, std::tuple_cat(arr)));
}

template <typename T, std::size_t I, auto Array, typename Types>
struct cartesian_value_type_product_impl;

/**
 * @brief Create the cartesian product of all @p Types together with the **single** value in @p Array.
 * @details Special case for a @p Array containing only a single element.
 * @tparam T the type of the elements in the @p Array
 * @tparam Array the std::array containing the value to create the cartesian product from
 * @tparam Types the types in the resulting cartesian product
 **/
template <typename T, const std::array<T, 1> *Array, typename... Types>
struct cartesian_value_type_product_impl<T, 1, Array, std::tuple<Types...>> {
    /// The cartesian value-type product of the value in @p Array and all types in @p Types.
    using type = std::tuple<parameter_definition<Types, std::get<0>(*Array)>...>;
};

/**
 * @brief Create the cartesian product of all @p Types together with the values in @p Array.
 * @tparam T the type of the elements in the @p Array
 * @tparam I the size of the @p Array
 * @tparam Array the std::array containing the value to create the cartesian product from
 * @tparam Types the types in the resulting cartesian product
 **/
template <typename T, std::size_t I, const std::array<T, I> *Array, typename... Types>
struct cartesian_value_type_product_impl<T, I, Array, std::tuple<Types...>> {
    static_assert(I > 1, "Cannot use an empty array for a value-type product!");

    /// The same as @p Array but with the first element removed.
    static constexpr std::array<T, I - 1> remaining_array = array_without_first_element(*Array);

    /// Create the value-type cartesian product, recursively.
    using type = plssvm::detail::concat_tuple_types_t<
        std::tuple<parameter_definition<Types, std::get<0>(*Array)>...>,
        typename cartesian_value_type_product_impl<T, I - 1, &remaining_array, std::tuple<Types...>>::type>;
};

}  // namespace detail

/**
 * @brief Create a value-type cartesian product between all values in @p Array and all types in @p Types.
 * @tparam Array the values
 * @tparam Types the types
 **/
template <auto Array, typename... Types>
using cartesian_value_type_product = detail::cartesian_value_type_product_impl<typename std::remove_pointer_t<decltype(Array)>::value_type, Array->size(), Array, Types...>;

/**
 * @brief A shorthand for the `typename cartesian_value_type_product::type`type.
 **/
template <auto Array, typename... Types>
using cartesian_value_type_product_t = typename cartesian_value_type_product<Array, Types...>::type;

/// A type list of all supported real types usable in google tests.
using real_type_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::real_type_list>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) usable in google tests.
using label_type_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::label_type_list>;

/// The cartesian product of all real types and label types usable in google test
using real_type_label_type_combination_gtest = detail::tuple_to_gtest_types_t<plssvm::detail::real_type_label_type_combination_list>;

/// A list of all available kernel function types.
constexpr std::array<plssvm::kernel_function_type, 3> kernel_functions_to_test{
    plssvm::kernel_function_type::linear, plssvm::kernel_function_type::polynomial, plssvm::kernel_function_type::rbf
};
/// A type list of all supported real_type and kernel_function_type combinations.
using real_type_kernel_function_list = cartesian_value_type_product_t<&kernel_functions_to_test, plssvm::detail::real_type_list>;
/// A type list of all supported real_type and kernel_function_type combinations usable in google tests.
using real_type_kernel_function_gtest = detail::tuple_to_gtest_types_t<real_type_kernel_function_list>;

/// A list of all available layout types.
constexpr std::array<plssvm::layout_type, 2> layout_types_to_test{
    plssvm::layout_type::aos, plssvm::layout_type::soa
};
/// A type list of all supported real_type and layout_type combinations.
using real_type_layout_type_list = cartesian_value_type_product_t<&layout_types_to_test, plssvm::detail::real_type_list>;
/// A type list of all supported real_type and layout_type combinations usable in google tests.
using real_type_layout_type_gtest = detail::tuple_to_gtest_types_t<real_type_layout_type_list>;

/// A list of all available classification types.
constexpr std::array<plssvm::classification_type, 2> classification_types_to_test{
    plssvm::classification_type::oaa, plssvm::classification_type::oao
};
/// A type list of all supported label_type and classification_type combinations.
using label_type_classification_type_list = cartesian_value_type_product_t<&classification_types_to_test, plssvm::detail::label_type_list>;
/// A type list of all supported label_type and classification_type combinations usable in google tests.
using label_type_classification_type_gtest = detail::tuple_to_gtest_types_t<label_type_classification_type_list>;

/// A type list of all supported label_type and layout_type combinations.
using label_type_layout_type_list = cartesian_value_type_product_t<&layout_types_to_test, plssvm::detail::label_type_list>;
/// A type list of all supported label_type and layout_type combinations usable in google tests.
using label_type_layout_type_gtest = detail::tuple_to_gtest_types_t<label_type_layout_type_list>;

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_