/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All possible `real_type`s and `label_type`s for a `plssvm::model` and `plssvm::data_set`.
 */

#ifndef PLSSVM_DETAIL_TYPE_LIST_HPP_
#define PLSSVM_DETAIL_TYPE_LIST_HPP_
#pragma once

#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::disjunction, std::is_same

namespace plssvm::detail {

/// A type list of all supported real types as `std::tuple`.
using supported_real_types = std::tuple<float, double>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using supported_label_types_classification = std::tuple<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double, std::string>;

/// A type list of a reduced number of supported label types as `std::tuple`.
using supported_label_types_classification_reduced = std::tuple<bool, int, double, std::string>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using supported_label_types_regression = std::tuple<short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double>;

/// A type list of a reduced number of supported label types as `std::tuple`.
using supported_label_types_regression_reduced = std::tuple<int, double>;

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Tuple the tuple type
 */
template <typename T, typename Tuple>
struct tuple_contains;

/**
 * @brief Checks whether the type @p T is present in the @p Types contained in a `std::tuple`.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Types the types in the tuple
 */
template <typename T, typename... Types>
struct tuple_contains<T, std::tuple<Types...>> : std::disjunction<std::is_same<T, Types>...> { };

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 */
template <typename T, typename Tuple>
inline constexpr bool tuple_contains_v = tuple_contains<T, Tuple>::value;

/**
 * @brief Checks whether the types in the tuple @p SubSetTuple are **all** contained in the tuple @p BaseSetTuple, i.e., @p SubSetTuple is a subset of @p BaseSetTuple.
 * @tparam SubSetTuple the tuple that should be a subset of @p BaseSetTuple
 * @tparam BaseSetTuple the base tuple
 */
template <typename SubSetTuple, typename BaseSetTuple>
struct tuple_subset_of;

/**
 * @brief Checks whether the @p SubSetTypes are **all** present in @p BaseSetTypes.
 * @tparam SubSetTypes the types that should be a subset
 * @tparam BaseSetTypes the base types
 */
template <typename... SubSetTypes, typename... BaseSetTypes>
struct tuple_subset_of<std::tuple<SubSetTypes...>, std::tuple<BaseSetTypes...>> : std::integral_constant<bool, ((tuple_contains_v<SubSetTypes, std::tuple<BaseSetTypes...>>) && ...)> { };

/**
 * @brief Checks whether @p SubSetTuple is a type subset of @p BaseSetTuple.
 */
template <typename SubSetTuple, typename BaseSetTuple>
inline constexpr bool tuple_subset_of_v = tuple_subset_of<SubSetTuple, BaseSetTuple>::value;


}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_TYPE_LIST_HPP_
