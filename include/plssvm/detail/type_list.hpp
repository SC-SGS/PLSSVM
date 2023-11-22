/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All possible `real_type`s and `label_type`s for a plssvm::model and plssvm::data_set.
 */

#ifndef PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
#define PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
#pragma once

#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::disjunction, std::is_same

namespace plssvm::detail {

/// A type list of all supported real types as `std::tuple`.
using supported_real_types = std::tuple<float, double>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using supported_label_types = std::tuple<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double, std::string>;

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 * @details Not implemented.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Tuple the tuple type
 */
template <typename T, typename Tuple>
struct tuple_contains;

/**
 * @brief Checks whether the type @p T is present in the tuple @p Types.
 * @tparam T the type to check if is contained in the tuple
 * @tparam Types the types in the tuple
 */
template <typename T, typename... Types>
struct tuple_contains<T, std::tuple<Types...>> : std::disjunction<std::is_same<T, Types>...> {};

/**
 * @brief Checks whether the type @p T is present in the @p Tuple.
 */
template <typename T, typename TypeList>
constexpr inline bool tuple_contains_v = tuple_contains<T, TypeList>::value;

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_TYPE_LIST_MANIPULATION_HPP_
