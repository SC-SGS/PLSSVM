/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines universal utility functions.
 */

#pragma once

#include "plssvm/default_value.hpp"  // plssvm::default_value

#include <cstddef>      // std::size_t
#include <map>          // std::map
#include <tuple>        // std::forward_as_tuple, std::get
#include <type_traits>  // std::underlying_type_t, std::is_enum_v

namespace plssvm::detail {

/**
 * @brief Type-dependent expression that always evaluates to `false`.
 */
template <typename>
constexpr bool always_false_v = false;

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

/**
 * @brief Get the @p I-th element of the parameter pack @p args.
 * @tparam I the index of the element to get
 * @tparam Types the types inside the parameter pack
 * @param[in] args the values of the parameter pack
 * @return the @p I-th element of @p args (`[[nodiscard]]`)
 */
template <std::size_t I, typename... Types>
[[nodiscard]] constexpr decltype(auto) get(Types &&...args) noexcept {
    static_assert(I < sizeof...(Types), "Out-of-bounce access!: too few elements in parameter pack");
    return std::get<I>(std::forward_as_tuple(args...));
}

/**
 * @brief Converts an enumeration to its underlying type.
 * @tparam Enum the enumeration type
 * @param[in] e enumeration value to convert
 * @return the integer value of the underlying type of `Enum`, converted from @p e (`[[nodiscard]]`)
 */
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(const Enum e) noexcept {
    static_assert(std::is_enum_v<Enum>, "e must be an enumeration type!");
    return static_cast<std::underlying_type_t<Enum>>(e);
}

template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(const default_value<Enum> &e) noexcept {
    return to_underlying(e.value());
}

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the container @p c.
 * @details See: https://en.cppreference.com/w/cpp/container/map/erase_if
 * @tparam Key the type of the map's key
 * @tparam T the map's value type
 * @tparam Compare the map's comparator type
 * @tparam Alloc the map's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the map
 * @param[in,out] c the map to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the map
 * @return the number of erased elements
 */
template <class Key, class T, class Compare, class Alloc, class Pred>
inline typename std::map<Key, T, Compare, Alloc>::size_type erase_if(std::map<Key, T, Compare, Alloc> &c, Pred pred) {
    auto old_size = c.size();
    for (auto i = c.begin(), last = c.end(); i != last;) {
        if (pred(*i)) {
            i = c.erase(i);
        } else {
            ++i;
        }
    }
    return old_size - c.size();
}

template <typename Key, typename T, typename Compare, typename Alloc>
[[nodiscard]] inline bool contains_key(const std::map<Key, T, Compare, Alloc> &map, const Key &key) {
    return map.count(key) > 0;
}

}  // namespace plssvm::detail