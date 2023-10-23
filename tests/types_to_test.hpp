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
    static constexpr std::tuple<decltype(Values)...> values{ Values... };
};

template <typename T>
struct is_value_list : std::false_type {};
template <auto... Values>
struct is_value_list<value_list<Values...>> : std::true_type {};
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
struct is_type_list : std::false_type {};
template <typename... Types>
struct is_type_list<type_list<Types...>> : std::true_type {};
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
//                                            cartesian product of up to three value arrays                                            //
//*************************************************************************************************************************************//

template <typename, std::size_t, std::size_t, auto, typename, std::size_t, std::size_t, auto, typename, std::size_t, std::size_t, auto>
struct cartesian_value_product_impl_3;

/**
 * @brief Terminate the recursion for the case with three value arrays.
 */
template <typename T, std::size_t I_SIZE, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, const std::array<U, J_SIZE> *Array2, typename V, std::size_t K_SIZE, const std::array<V, K_SIZE> *Array3>
struct cartesian_value_product_impl_3<T, I_SIZE, 0, Array1, U, J_SIZE, 0, Array2, V, K_SIZE, 0, Array3> {
    using type = std::tuple<value_list<std::get<0>(*Array1), std::get<0>(*Array2), std::get<0>(*Array3)>>;
};

/**
 * @brief Recursion case: the indices of the 2nd and 3rd arrays are zero -> reset their indices and decrement the index of the 1st array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, const std::array<U, J_SIZE> *Array2, typename V, std::size_t K_SIZE, const std::array<V, K_SIZE> *Array3>
struct cartesian_value_product_impl_3<T, I_SIZE, I, Array1, U, J_SIZE, 0, Array2, V, K_SIZE, 0, Array3> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array1), std::get<0>(*Array2), std::get<0>(*Array3)>>,
        typename cartesian_value_product_impl_3<T, I_SIZE, I - 1, Array1, U, J_SIZE, J_SIZE - 1, Array2, V, K_SIZE, K_SIZE - 1, Array3>::type>;
};

/**
 * @brief Recursion case: the index of the 3rd array is zero -> reset its index and decrement the index of the 2nd array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, std::size_t J, const std::array<U, J_SIZE> *Array2, typename V, std::size_t K_SIZE, const std::array<V, K_SIZE> *Array3>
struct cartesian_value_product_impl_3<T, I_SIZE, I, Array1, U, J_SIZE, J, Array2, V, K_SIZE, 0, Array3> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array1), std::get<J>(*Array2), std::get<0>(*Array3)>>,
        typename cartesian_value_product_impl_3<T, I_SIZE, I, Array1, U, J_SIZE, J - 1, Array2, V, K_SIZE, K_SIZE - 1, Array3>::type>;
};

/**
 * @brief General case for three value arrays: decrement the index of the 3rd array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, std::size_t J, const std::array<U, J_SIZE> *Array2, typename V, std::size_t K_SIZE, std::size_t K, const std::array<V, K_SIZE> *Array3>
struct cartesian_value_product_impl_3<T, I_SIZE, I, Array1, U, J_SIZE, J, Array2, V, K_SIZE, K, Array3> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array1), std::get<J>(*Array2), std::get<K>(*Array3)>>,
        typename cartesian_value_product_impl_3<T, I_SIZE, I, Array1, U, J_SIZE, J, Array2, V, K_SIZE, K - 1, Array3>::type>;
};

template <typename, std::size_t, std::size_t, auto, typename, std::size_t, std::size_t, auto>
struct cartesian_value_product_impl_2;

/**
 * @brief Terminate the recursion for the case with two value arrays.
 */
template <typename T, std::size_t I_SIZE, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, const std::array<U, J_SIZE> *Array2>
struct cartesian_value_product_impl_2<T, I_SIZE, 0, Array1, U, J_SIZE, 0, Array2> {
    using type = std::tuple<value_list<std::get<0>(*Array1), std::get<0>(*Array2)>>;
};

/**
 * @brief Recursion case: the index of the 2nd array is zero -> reset its index and decrement the index of the 1st array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, const std::array<U, J_SIZE> *Array2>
struct cartesian_value_product_impl_2<T, I_SIZE, I, Array1, U, J_SIZE, 0, Array2> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array1), std::get<0>(*Array2)>>,
        typename cartesian_value_product_impl_2<T, I_SIZE, I - 1, Array1, U, J_SIZE, J_SIZE - 1, Array2>::type>;
};

/**
 * @brief General case for two value arrays: decrement the index of the 2nd array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array1, typename U, std::size_t J_SIZE, std::size_t J, const std::array<U, J_SIZE> *Array2>
struct cartesian_value_product_impl_2<T, I_SIZE, I, Array1, U, J_SIZE, J, Array2> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array1), std::get<J>(*Array2)>>,
        typename cartesian_value_product_impl_2<T, I_SIZE, I, Array1, U, J_SIZE, J - 1, Array2>::type>;
};

template <typename, std::size_t, std::size_t, auto>
struct cartesian_value_product_impl_1;

/**
 * @brief Terminate the recursion for the case with one value array.
 */
template <typename T, std::size_t I_SIZE, const std::array<T, I_SIZE> *Array>
struct cartesian_value_product_impl_1<T, I_SIZE, 0, Array> {
    using type = std::tuple<value_list<std::get<0>(*Array)>>;
};

/**
 * @brief General case for one value array: decrement the index of the 1st array by one.
 */
template <typename T, std::size_t I_SIZE, std::size_t I, const std::array<T, I_SIZE> *Array>
struct cartesian_value_product_impl_1<T, I_SIZE, I, Array> {
    using type = concat_tuple_types_t<
        std::tuple<value_list<std::get<I>(*Array)>>,
        typename cartesian_value_product_impl_1<T, I_SIZE, I - 1, Array>::type>;
};

//*************************************************************************************************************************************//
//               cartesian product dispatcher to select the correct function based on the number of provided value arrays              //
//*************************************************************************************************************************************//

// TODO: https://godbolt.org/z/bhEPfW17r

template <std::size_t SIZE, auto... Arrays>
struct cartesian_value_product_dispatcher;

/**
 * @brief Only one value array is given -> dispatch to `cartesian_value_product_impl_1`.
 */
template <auto Array>
struct cartesian_value_product_dispatcher<1, Array> {
    using type = typename cartesian_value_product_impl_1<
        typename std::remove_pointer_t<decltype(Array)>::value_type,
        Array->size(),
        Array->size() - 1,
        Array>::type;
};

/**
 * @brief Two value arrays are given -> dispatch to `cartesian_value_product_impl_2`.
 */
template <auto Array1, auto Array2>
struct cartesian_value_product_dispatcher<2, Array1, Array2> {
    using type = typename cartesian_value_product_impl_2<
        typename std::remove_pointer_t<decltype(Array1)>::value_type,
        Array1->size(),
        Array1->size() - 1,
        Array1,
        typename std::remove_pointer_t<decltype(Array2)>::value_type,
        Array2->size(),
        Array2->size() - 1,
        Array2>::type;
};

/**
 * @brief Three value arrays are given -> dispatch to `cartesian_value_product_impl_3`.
 */
template <auto Array1, auto Array2, auto Array3>
struct cartesian_value_product_dispatcher<3, Array1, Array2, Array3> {
    using type = typename cartesian_value_product_impl_3<
        typename std::remove_pointer_t<decltype(Array1)>::value_type,
        Array1->size(),
        Array1->size() - 1,
        Array1,
        typename std::remove_pointer_t<decltype(Array2)>::value_type,
        Array2->size(),
        Array2->size() - 1,
        Array2,
        typename std::remove_pointer_t<decltype(Array3)>::value_type,
        Array3->size(),
        Array3->size() - 1,
        Array3>::type;
};

//*************************************************************************************************************************************//
//                                                    cartesian product of two types                                                   //
//*************************************************************************************************************************************//

// calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding type combinations stored in a WrapperType.
template <template <typename...> typename WrapperType, typename S, typename T>
struct cartesian_type_product_impl;

/**
 * @brief Create the cartesian product of the types given in two std::tuple.
 * @tparam WrapperType the type to store the type combinations to
 * @tparam FirstTupleType the first type in the first std::tuple
 * @tparam FirstTupleRemainingTypes  the remaining types in the first tuple
 * @tparam SecondTupleTypes the types in the second tuple
 */
template <template <typename...> typename WrapperType, typename FirstTupleType, typename... FirstTupleRemainingTypes, typename... SecondTupleTypes>
struct cartesian_type_product_impl<WrapperType, std::tuple<FirstTupleType, FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>> {
    // the cartesian product of {FirstTupleType} and {SecondTupleTypes...} is a list of real_type_label_type_combination
    using FirstTupleType_cross_SecondTupleTypes = std::tuple<WrapperType<FirstTupleType, SecondTupleTypes>...>;

    // the cartesian product of {FirstTupleRemainingTypes...} and {Ts...} (computed recursively)
    using FirstTupleRemainingTypes_cross_SecondTupleTypes = typename cartesian_type_product_impl<WrapperType, std::tuple<FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>>::type;

    // concatenate both products
    using type = concat_tuple_types_t<FirstTupleType_cross_SecondTupleTypes, FirstTupleRemainingTypes_cross_SecondTupleTypes>;
};

/**
 * @brief End the recursion if the first tuple does not have any types left.
 * @tparam WrapperType the type to store the type combinations to
 * @tparam SecondTupleTypes the types in the second tuple
 */
template <template <typename...> typename WrapperType, typename... SecondTupleTypes>
struct cartesian_type_product_impl<WrapperType, std::tuple<>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<>;  // the cartesian product of {}x{...} is {}
};

//*************************************************************************************************************************************//
//                         be able to create a test_parameter even if no type- or value-list has been provided                         //
//*************************************************************************************************************************************//

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
    // clang-format off
    using type = std::conditional_t<
        (is_type_list_v<Types> && ...),  // check whether only type_lists have been provided
            typename wrap_types_in_test_parameter_with_empty_value_list<std::tuple<Types...>>::type,  // true -> create test_parameters
            std::conditional_t<  // false
                (is_value_list_v<Types> && ...),  // check whether only value_lists have been provided
                   typename wrap_types_in_test_parameter_with_empty_type_list<std::tuple<Types...>>::type,  // true -> create test_parameters
                   void  // should be unreachable
            >
        >;
    // clang-format on
};

/**
 * @brief Create `test_parameter`s with a type- and value-list.
 */
template <typename T, typename U>
struct create_test_parameters_dispatcher<2, T, U> {
    using type = typename detail::cartesian_type_product_impl<test_parameter, T, U>::type;
};

}  // namespace detail

//*************************************************************************************************************************************//
//                                               simple wrappers for a single std::tuple                                               //
//*************************************************************************************************************************************//

template <typename T>
struct wrap_tuple_types_in_type_lists;

/**
 * @brief Wrap all types in the std::tuple in a `type_list`.
 */
template <typename... Types>
struct wrap_tuple_types_in_type_lists<std::tuple<Types...>> {
    using type = std::tuple<type_list<Types>...>;
};
/**
 * @brief Shorthand for `typename wrap_tuple_types_in_type_lists<T>::type`.
 */
template <typename T>
using wrap_tuple_types_in_type_lists_t = typename wrap_tuple_types_in_type_lists<T>::type;

/**
 * @brief Wrap all values in the std::array in a `value_list`.
 */
template <auto Array>
using wrap_value_array_in_value_lists = detail::cartesian_value_product_impl_1<
    typename std::remove_pointer_t<decltype(Array)>::value_type,
    Array->size(),
    Array->size() - 1,
    Array>;
/**
 * @brief Shorthand for `typename wrap_value_array_in_value_lists<Array>::type`.
 */
template <auto Array>
using wrap_value_array_in_value_lists_t = typename wrap_value_array_in_value_lists<Array>::type;

//*************************************************************************************************************************************//
//                                                          helper shorthands                                                          //
//*************************************************************************************************************************************//

/**
 * @brief Create a std::tuple of `value_list`s as the result of the cartesian product of the values in up to three arrays.
 */
template <auto... Arrays>
using cartesian_value_product = detail::cartesian_value_product_dispatcher<sizeof...(Arrays), Arrays...>;
/**
 * @brief Shorthand for `typename cartesian_value_product<...>::type`.
 */
template <auto... Arrays>
using cartesian_value_product_t = typename cartesian_value_product<Arrays...>::type;

/**
 * @brief Create a std::tuple of `type_list`s as the result of the cartesian product of the two types.
 */
template <typename... Types>
using cartesian_type_product = detail::cartesian_type_product_impl<type_list, Types...>;
/**
 * @brief Shorthand for `typename cartesian_type_product<...>::type`.
 */
template <typename... Types>
using cartesian_type_product_t = typename cartesian_type_product<Types...>::type;

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
constexpr std::array<plssvm::kernel_function_type, 3> kernel_functions_to_test{
    plssvm::kernel_function_type::linear, plssvm::kernel_function_type::polynomial, plssvm::kernel_function_type::rbf
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
constexpr std::array<plssvm::solver_type, 4> solver_types_to_test = {
    plssvm::solver_type::automatic, plssvm::solver_type::cg_explicit, plssvm::solver_type::cg_streaming, plssvm::solver_type::cg_implicit
};

/// A list of all solver types.
using solver_type_list = wrap_value_array_in_value_lists_t<&solver_types_to_test>;
/// A list of all kernel function types.
using kernel_function_type_list = wrap_value_array_in_value_lists_t<&kernel_functions_to_test>;
/// A list of all classification types.
using classification_type_list = wrap_value_array_in_value_lists_t<&classification_types_to_test>;
/// A list of all layout types.
using layout_type_list = wrap_value_array_in_value_lists_t<&layout_types_to_test>;
/// A list of a combination of all solver and kernel function types.
using solver_and_kernel_function_type_list = cartesian_value_product_t<&solver_types_to_test, &kernel_functions_to_test>;
/// A list of a combination of all kernel function and classification types.
using kernel_function_and_classification_type_list = cartesian_value_product_t<&kernel_functions_to_test, &classification_types_to_test>;
/// A list of a combination of all solver, kernel function, and classification types.
using solver_and_kernel_function_and_classification_type_list = cartesian_value_product_t<&solver_types_to_test, &kernel_functions_to_test, &classification_types_to_test>;

/// A list of all supported real types based on `plssvm::detail::supported_real_types`.
using real_type_list = wrap_tuple_types_in_type_lists_t<plssvm::detail::supported_real_types>;
/// A list of all supported label types based on `plssvm::detail::supported_label_types`.
using label_type_list = wrap_tuple_types_in_type_lists_t<plssvm::detail::supported_label_types>;

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
/// A list of a combination of all supported label types and classification, kernel function, and solver types wrapped in a Google test type.
using label_type_solver_and_kernel_function_and_classification_type_gtest = combine_test_parameters_gtest_t<label_type_list, solver_and_kernel_function_and_classification_type_list>;

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_