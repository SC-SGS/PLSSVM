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
#include "plssvm/detail/type_list.hpp"       // plssvm::detail::{real_type_list, label_type_list}
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::layout_type
#include "plssvm/solver_types.hpp"           // plssvm::solver_type

#include "gtest/gtest.h"  // ::testing::Types

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <tuple>        // std::tuple, std::tuple_element_t, std::get, std::array
#include <type_traits>  // std::true_type, std::false_type, std::remove_pointer_t, std::conditional_t

namespace util {

//*************************************************************************************************************************************//
//                                    helper structs representing a compile-time type- and value-list                                  //
//*************************************************************************************************************************************//

/**
 * @brief A wrapper for a compile-time list of constant values.
 * @tparam Values the values
 */
template <auto... Values>
struct value_list {
    /// The compile-time constant values wrapped in a std::tuple.
    constexpr static std::tuple<decltype(Values)...> values{ Values... };
};

template <typename T>
struct is_value_list : std::false_type { };

template <auto... Values>
struct is_value_list<value_list<Values...>> : std::true_type { };

/**
 * @brief Check whether @p T is a value_list independent of its possible compile-time constants.
 * @tparam T the type to check
 */
template <typename T>
constexpr bool is_value_list_v = is_value_list<T>::value;

/**
 * @brief A wrapper for a compile-time list of types.
 * @tparam Types the types
 */
template <typename... Types>
struct type_list {
    /// The compile-time types wrapped in a std::tuple.
    using types = std::tuple<Types...>;
};

template <typename T>
struct is_type_list : std::false_type { };

template <typename... Types>
struct is_type_list<type_list<Types...>> : std::true_type { };

/**
 * @brief Check whether @p T s a type_list independent fof its possible compile-time types.
 * @tparam T the type to check
 */
template <typename T>
constexpr bool is_type_list_v = is_type_list<T>::value;

/**
 * @brief A struct encapsulating a compile-time type_list and a value_list; used in all TYPED_TESTs for consistencies.
 * @tparam Types the type_list
 * @tparam Values the value_list
 */
template <typename Types, typename Values>
struct test_parameter {
    static_assert(is_type_list_v<Types>, "Error: 'Types' isn't of type 'type_list'!");
    static_assert(is_value_list_v<Values>, "Error: 'Values' isn't of type 'value_list'!");

    /// The compile-time type_list.
    using types = Types;
    /// The compile-time value_list.
    using values = Values;
};

/**
 * @brief Get the type of the type_list in @p TestParameter at the position @p I.
 * @tparam I the position of the type to retrieve
 * @tparam TestParameter the type of the test_parameter
 */
template <std::size_t I, typename TestParameter, typename TypeList = typename TestParameter::types>
using test_parameter_type_at_t = std::tuple_element_t<I, typename TypeList::types>;
/**
 * @brief Get the value of the value_list in @p TestParameter at the position @p I.
 * @tparam I the position of the value to retrieve
 * @tparam TestParameter the type of the test_parameter
 */
template <std::size_t I, typename TestParameter, typename ValueList = typename TestParameter::values>
constexpr auto test_parameter_value_at_v = std::get<I>(ValueList::values);

//*************************************************************************************************************************************//
//                                                        implementation details                                                       //
//*************************************************************************************************************************************//

namespace detail {

/**
 * @brief Copy all types in the @p Tuple using the indices @p Is.
 * @tparam Tuple the tuple types to clone
 * @tparam Is the index sequence used to clone the tuple types
 * @param[in] tuple the tuple to clone
 */
template <typename Tuple, std::size_t... Is>
[[nodiscard]] constexpr auto expand_tuple(std::index_sequence<Is...>, const Tuple &tuple) {
    return std::tuple_cat((static_cast<void>(Is), tuple)...);
}

/**
 * @brief Copy all types in the @p Tuple @p N times.
 * @tparam N the time how often the types should be cloned
 * @tparam Tuple the tuple types to clone
 */
template <std::size_t N, typename Tuple>
struct clone_tuple_types {
    using type = decltype(expand_tuple(std::make_index_sequence<N>(), std::declval<Tuple>()));
};

/**
 * @brief Shorthand for the `typename clone_tuple_types<N, Tuple>::type` type.
 */
template <std::size_t N, typename Tuple>
using clone_tuple_types_t = typename clone_tuple_types<N, Tuple>::type;

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

template <typename S, typename T>
struct concat_tuple_types;

/**
 * @brief Concatenate the types of two tuples.
 * @tparam FirstTupleTypes the types in the first tuple
 * @tparam SecondTupleTypes the types in the second tuple
 */
template <typename... FirstTupleTypes, typename... SecondTupleTypes>
struct concat_tuple_types<std::tuple<FirstTupleTypes...>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<FirstTupleTypes..., SecondTupleTypes...>;
};

/**
 * @brief Shorthand for `typename util::detail::concat_tuple_types<...>::types`.
 */
template <typename... T>
using concat_tuple_types_t = typename concat_tuple_types<T...>::type;

//*************************************************************************************************************************************//
//                                          cartesian product of an arbitrary number of arrays                                         //
//*************************************************************************************************************************************//

template <typename T, auto N>
struct add_to_value_list;

/**
 * @brief Add the compile-time constant value @p N to the values in the provided `value_list`.
 * @tparam Values the values already in the `value_list`
 * @tparam N the new value to be added to the `value_list`
 */
template <auto... Values, auto N>
struct add_to_value_list<value_list<Values...>, N> {
    using type = value_list<N, Values...>;
};

/**
 * @brief Shorthand for `typename add_to_value_list<...>::type`.
 */
template <typename T, auto N>
using add_to_value_list_t = typename add_to_value_list<T, N>::type;

template <const auto &Array, typename Indices = std::make_index_sequence<Array.size()>>
struct wrap_in_value_list;

/**
 * @brief Wrap the elements in @p Array in `value_list` types and return a std::tuple to this type.
 * @tparam Array the values to wrap
 * @tparam I index sequence used to iterate over the @p Array
 */
template <const auto &Array, std::size_t... I>
struct wrap_in_value_list<Array, std::index_sequence<I...>> {
    using type = std::tuple<value_list<std::get<I>(Array)>...>;
};

/**
 * @brief Shorthand for `typename wrap_in_value_list<...>::type`.
 */
template <const auto &Array>
using wrap_in_value_list_t = typename wrap_in_value_list<Array>::type;

template <typename, const auto &, typename>
struct combine_values_impl { };

/**
 * @brief Iteratively add the values in @p Array at position @p IS to the `value_list`s in the std::tuple @p Tuple.
 * @tparam Tuple the tuple types
 * @tparam Array the array values to add
 * @tparam Is the indices in @p Array
 */
template <typename Tuple, const auto &Array, std::size_t... Is>
struct combine_values_impl<Tuple, Array, std::index_sequence<Is...>> {
    constexpr static std::size_t N = Array.size();
    using type = std::tuple<add_to_value_list_t<std::tuple_element_t<Is, Tuple>, std::get<Is % N>(Array)>...>;
};

/**
 * @brief Add the values in @p Array to the `value_list`s in the std::tuple @p Tuple.
 * @tparam Tuple the tuple types
 * @tparam Array the array values to add
 */
template <typename Tuple, const auto &Array>
struct combine_values {
    // clone the types in the Tuple N-times where N is the number of values in the Array
    using cloned_tuple = clone_tuple_types_t<Array.size(), Tuple>;
    using type = typename combine_values_impl<cloned_tuple, Array, std::make_index_sequence<std::tuple_size_v<cloned_tuple>>>::type;
};

/**
 * @brief Shorthand for `typename combine_values<...>::type`.
 */
template <const auto &Array, typename Tuple>
using combine_values_t = typename combine_values<Tuple, Array>::type;

/**
 * @brief Calculate the cartesian product of the values in @p FirstArray and @p RemainingArrays recursively.
 * @tparam FirstArray the first array to combine
 * @tparam RemainingArrays the remaining arrays
 */
template <const auto &FirstArray, const auto &...RemainingArrays>
struct cartesian_value_product {
    using type = combine_values_t<FirstArray, typename cartesian_value_product<RemainingArrays...>::type>;
};

/**
 * @brief Recursion termination: the cartesian product of a single array is the array itself wrapped in `value_list`s.
 * @tparam Array the array to wrap
 */
template <const auto &Array>
struct cartesian_value_product<Array> {
    using type = wrap_in_value_list_t<Array>;
};

//*************************************************************************************************************************************//
//                               cartesian product of an arbitrary number of types wrapped in std::tuple                               //
//*************************************************************************************************************************************//

template <typename T, typename U>
struct add_to_type_list;

/**
 * @brief Add the type @p U to the types in the provided `type_list`.
 * @tparam Types the types already in the `type_list`
 * @tparam U the new type to be added to the `type_list`
 */
template <typename... Types, typename U>
struct add_to_type_list<type_list<Types...>, U> {
    using type = type_list<U, Types...>;
};

/**
 * @brief Shorthand for `typename add_to_type_list<...>::type`.
 */
template <typename T, typename U>
using add_to_type_list_t = typename add_to_type_list<T, U>::type;

template <typename Tuple>
struct wrap_in_type_list;

/**
 * @brief Wrap the @p Types given in a std::tuple in `type_list` types and return a std::tuple to this type.
 * @tparam Types the types to wrap
 */
template <typename... Types>
struct wrap_in_type_list<std::tuple<Types...>> {
    using type = std::tuple<type_list<Types>...>;
};

/**
 * @brief Shorthand for `typename wrap_in_type_list<...>::type`.
 */
template <typename Tuple>
using wrap_in_type_list_t = typename wrap_in_type_list<Tuple>::type;

template <typename Tuple, typename CurrentTuple, typename>
struct combine_types_impl { };

/**
 * @brief Iteratively add the types in @p CurrentTuple at position @p IS to the `type_list`s in the std::tuple @p Tuple.
 * @tparam Tuple the tuple types
 * @tparam CurrentTuple the types in the current tuple
 * @tparam Is the indices in @p Array
 */
template <typename Tuple, typename CurrentTuple, std::size_t... Is>
struct combine_types_impl<Tuple, CurrentTuple, std::index_sequence<Is...>> {
    constexpr static std::size_t N = std::tuple_size_v<CurrentTuple>;
    using type = std::tuple<add_to_type_list_t<std::tuple_element_t<Is, Tuple>, std::tuple_element_t<Is % N, CurrentTuple>>...>;
};

/**
 * @brief Add the types in @p CurrentTuple to the `type_list`s in the std::tuple @p Tuple.
 * @tparam Tuple the tuple types
 * @tparam CurrentTuple the types in the current tuple
 */
template <typename Tuple, typename CurrentTuple>
struct combine_types {
    // clone the types in the Tuple N-times where N is the number of types in the CurrentTuple
    using cloned_tuple = clone_tuple_types_t<std::tuple_size_v<CurrentTuple>, Tuple>;
    using type = typename combine_types_impl<cloned_tuple, CurrentTuple, std::make_index_sequence<std::tuple_size_v<cloned_tuple>>>::type;
};

/**
 * @brief Shorthand for `typename combine_types<...>::type`.
 */
template <typename Tuple, typename ResultTuple>
using combine_types_t = typename combine_types<ResultTuple, Tuple>::type;

/**
 * @brief Calculate the cartesian product of the types in @p FirstTuple and @p RemainingTuples recursively.
 * @tparam FirstTuple the first std:tuple to combine
 * @tparam RemainingTuples the remaining std::tuple
 */
template <typename FirstTuple, typename... RemainingTuples>
struct cartesian_type_product {
    using type = combine_types_t<FirstTuple, typename cartesian_type_product<RemainingTuples...>::type>;
};

/**
 * @brief Recursion termination: the cartesian product of a single tuple is the tuple itself wrapped in `type_list`s.
 * @tparam Tuple the tuple to wrap
 */
template <typename Tuple>
struct cartesian_type_product<Tuple> {
    using type = wrap_in_type_list_t<Tuple>;
};

//*************************************************************************************************************************************//
//                         be able to create a test_parameter even if no type- or value-list has been provided                         //
//*************************************************************************************************************************************//

// calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding type combinations stored in a WrapperType.
template <typename S, typename T>
struct create_test_parameters_impl;

/**
 * @brief Create the cartesian product of the types given in two std::tuple.
 * @tparam FirstTupleType the first type in the first std::tuple
 * @tparam FirstTupleRemainingTypes  the remaining types in the first tuple
 * @tparam SecondTupleTypes the types in the second tuple
 */
template <typename FirstTupleType, typename... FirstTupleRemainingTypes, typename... SecondTupleTypes>
struct create_test_parameters_impl<std::tuple<FirstTupleType, FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>> {
    // the cartesian product of {FirstTupleType} and {SecondTupleTypes...} is a list of real_type_label_type_combination
    using FirstTupleType_cross_SecondTupleTypes = std::tuple<test_parameter<FirstTupleType, SecondTupleTypes>...>;

    // the cartesian product of {FirstTupleRemainingTypes...} and {Ts...} (computed recursively)
    using FirstTupleRemainingTypes_cross_SecondTupleTypes = typename create_test_parameters_impl<std::tuple<FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>>::type;

    // concatenate both products
    using type = concat_tuple_types_t<FirstTupleType_cross_SecondTupleTypes, FirstTupleRemainingTypes_cross_SecondTupleTypes>;
};

/**
 * @brief End the recursion if the first tuple does not have any types left.
 * @tparam SecondTupleTypes the types in the second tuple
 */
template <typename... SecondTupleTypes>
struct create_test_parameters_impl<std::tuple<>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<>;  // the cartesian product of {}x{...} is {}
};

template <typename T>
struct wrap_types_in_test_parameter_with_empty_value_list;

/**
 * @brief Wrap the `type_list`s in a `test_parameter` using an empty `value_list`.
 */
template <typename... Types>
struct wrap_types_in_test_parameter_with_empty_value_list<std::tuple<Types...>> {
    // static_assert((is_type_list_v<Types> && ...), "Error: 'Types' doesn't only contain 'type_list's!");
    using type = std::tuple<test_parameter<Types, value_list<>>...>;
};

template <typename T>
struct wrap_types_in_test_parameter_with_empty_type_list;

/**
 * @brief Wrap the `value_list`s in a `test_parameter` using an empty `type_list`.
 */
template <typename... Types>
struct wrap_types_in_test_parameter_with_empty_type_list<std::tuple<Types...>> {
    // static_assert((is_value_list_v<Types> && ...), "Error: 'Types' doesn't only contain 'value_list's!");
    using type = std::tuple<test_parameter<type_list<>, Types>...>;
};

template <std::size_t SIZE, typename... T>
struct create_test_parameters_dispatcher;

/**
 * @brief Correctly create `test_parameter`s even if no type- or value-list has been provided.
 */
template <typename... Types>
struct create_test_parameters_dispatcher<1, std::tuple<Types...>> {
    using type = std::conditional_t<
        (is_type_list_v<Types> && ...),                                                              // check whether only type_lists have been provided
        typename wrap_types_in_test_parameter_with_empty_value_list<std::tuple<Types...>>::type,     // true -> create test_parameters
        std::conditional_t<                                                                          // false
            (is_value_list_v<Types> &&...),                                                          // check whether only value_lists have been provided
            typename wrap_types_in_test_parameter_with_empty_type_list<std::tuple<Types...>>::type,  // true -> create test_parameters
            void                                                                                     // should be unreachable
            >>;
};

/**
 * @brief Create `test_parameter`s with a type- and value-list.
 */
template <typename T, typename U>
struct create_test_parameters_dispatcher<2, T, U> {
    using type = typename detail::create_test_parameters_impl<T, U>::type;
};

}  // namespace detail

//*************************************************************************************************************************************//
//                                                          helper shorthands                                                          //
//*************************************************************************************************************************************//

/**
 * @brief Shorthand for `typename detail::cartesian_value_product<...>::type`.
 */
template <const auto &...Arrays>
using cartesian_value_product_t = typename detail::cartesian_value_product<Arrays...>::type;
/**
 * @brief Shorthand for `typename detail::cartesian_type_product<...>::type`.
 */
template <typename... Tuples>
using cartesian_type_product_t = typename detail::cartesian_type_product<Tuples...>::type;

/**
 * @brief Combine a std::tuple of `type_list`s and/or a std::tuple of `value_list`s to a std::tuple of `test_parameter`.
 */
template <typename... Types>
using combine_test_parameters_gtest = detail::tuple_to_gtest_types<typename detail::create_test_parameters_dispatcher<sizeof...(Types), Types...>::type>;
/**
 * @brief Shorthand for `typename combine_test_parameters_gtest<...>::type`.
 */
template <typename... Types>
using combine_test_parameters_gtest_t = typename combine_test_parameters_gtest<Types...>::type;

//*************************************************************************************************************************************//
//                                                          actual test lists                                                          //
//*************************************************************************************************************************************//

/// A list of all available kernel function types.
constexpr std::array<plssvm::kernel_function_type, 6> kernel_functions_to_test{
    plssvm::kernel_function_type::linear, plssvm::kernel_function_type::polynomial, plssvm::kernel_function_type::rbf,
    plssvm::kernel_function_type::sigmoid, plssvm::kernel_function_type::laplacian, plssvm::kernel_function_type::chi_squared
};
/// A list of all available layout types.
constexpr std::array<plssvm::layout_type, 2> layout_types_to_test{
    plssvm::layout_type::aos, plssvm::layout_type::soa
};
/// A list of all available classification types.
constexpr std::array<plssvm::classification_type, 2> classification_types_to_test{
    plssvm::classification_type::oaa, plssvm::classification_type::oao
};
/// A list of all available solver types.
constexpr std::array<plssvm::solver_type, 3> solver_types_to_test{
    plssvm::solver_type::automatic, plssvm::solver_type::cg_explicit, plssvm::solver_type::cg_implicit
};

/// A list of all solver types.
using solver_type_list = cartesian_value_product_t<solver_types_to_test>;
/// A list of all kernel function types.
using kernel_function_type_list = cartesian_value_product_t<kernel_functions_to_test>;
/// A list of all classification types.
using classification_type_list = cartesian_value_product_t<classification_types_to_test>;
/// A list of all layout types.
using layout_type_list = cartesian_value_product_t<layout_types_to_test>;
/// A list of a combination of all solver and kernel function types.
using solver_and_kernel_function_type_list = cartesian_value_product_t<solver_types_to_test, kernel_functions_to_test>;
/// A list of a combination of all kernel function and classification types.
using kernel_function_and_classification_type_list = cartesian_value_product_t<kernel_functions_to_test, classification_types_to_test>;
/// A list of a combination of all solver, kernel function, and classification types.
using solver_and_kernel_function_and_classification_type_list = cartesian_value_product_t<solver_types_to_test, kernel_functions_to_test, classification_types_to_test>;

/// A list of all supported real types based on `plssvm::detail::supported_real_types`.
using real_type_list = cartesian_type_product_t<plssvm::detail::supported_real_types>;

/// A list of all supported label types based on `plssvm::detail::supported_label_types`.
#if defined(PLSSVM_TEST_WITH_REDUCED_LABEL_TYPES)
using label_type_list = cartesian_type_product_t<plssvm::detail::supported_label_types_reduced>;
#else
using label_type_list = cartesian_type_product_t<plssvm::detail::supported_label_types>;
#endif

/// A list of all supported real types wrapped in a Google test type.
using real_type_gtest = combine_test_parameters_gtest_t<real_type_list>;
/// A list of all supported label types (currently arithmetic types and `std::string`) wrapped in a Google test type.
using label_type_gtest = combine_test_parameters_gtest_t<label_type_list>;
/// A list of a combination of all supported real types and layout types wrapped in a Google test type.
using real_type_layout_type_gtest = combine_test_parameters_gtest_t<real_type_list, layout_type_list>;
/// A list of a combination of all supported label types and classification types wrapped in a Google test type.
using label_type_classification_type_gtest = combine_test_parameters_gtest_t<label_type_list, classification_type_list>;
/// A list of a combination of all supported label types and layout types wrapped in a Google test type.
using label_type_layout_type_gtest = combine_test_parameters_gtest_t<label_type_list, layout_type_list>;
/// A list of a combination of all supported label types, kernel function and classification types wrapped in a Google test type.
using label_type_kernel_function_and_classification_type_gtest = combine_test_parameters_gtest_t<label_type_list, kernel_function_and_classification_type_list>;
/// A list of a combination of all supported label types and classification, kernel function, and solver types wrapped in a Google test type.
using label_type_solver_and_kernel_function_and_classification_type_gtest = combine_test_parameters_gtest_t<label_type_list, solver_and_kernel_function_and_classification_type_list>;

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_
