/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All possible `real_type` and `label_type` combinations for a plssvm::model and plssvm::data_set.
 */

#ifndef PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
#define PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
#pragma once

#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::disjunction, std::is_same_v

namespace plssvm::detail {

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

/// A type list of all supported real types as `std::tuple`.
using real_type_list = std::tuple<float, double>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using label_type_list = std::tuple<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double, std::string>;

/// The cartesian product of all real types and label types as `std::tuple`.
using real_type_label_type_combination_list = detail::cartesian_type_product_t<real_type_list, label_type_list>;

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 * @details Not implemented.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Tuple the tuple type
 */
template <typename T, typename Tuple>
struct type_list_contains;

/**
 * @brief Checks whether the type @p T is present in the tuple @p Types.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Types the types in the tuple
 */
template <typename T, typename... Types>
struct type_list_contains<T, std::tuple<Types...>> : std::disjunction<std::is_same<T, Types>...> {};

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 */
template <typename T, typename Tuple>
constexpr inline bool type_list_contains_v = type_list_contains<T, Tuple>::value;

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
