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

#ifndef PLSSVM_DETAIL_UTILITY_HPP_
#define PLSSVM_DETAIL_UTILITY_HPP_
#pragma once

#include "plssvm/default_value.hpp"  // plssvm::default_value

#include <algorithm>      // std::remove_if
#include <cstddef>        // std::size_t
#include <iterator>       // std::distance
#include <map>            // std::map
#include <set>            // std::set
#include <tuple>          // std::forward_as_tuple, std::get
#include <type_traits>    // std::remove_cv_t, std::remove_reference_t, std::underlying_type_t, std::is_enum_v
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector

namespace plssvm::detail {

/**
 * @brief Type-dependent expression that always evaluates to `false`.
 */
template <typename>
constexpr bool always_false_v = false;

/**
 * @brief Remove the topmost reference- and cv-qualifiers.
 * @details For more information see [`std::remove_cvref_t`](https://en.cppreference.com/w/cpp/types/remove_cvref).
 */
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

/**
 * @brief Converts an enumeration wrapped in a plssvm::default_value to its underlying type.
 * @tparam Enum the enumeration type
 * @param[in] e enumeration value to convert wrapped in a plssvm::default_value
 * @return the integer value of the underlying type of `Enum`, converted from @p e (`[[nodiscard]]`)
 */
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(const default_value<Enum> &e) noexcept {
    static_assert(std::is_enum_v<Enum>, "e must be an enumeration type!");
    return to_underlying(e.value());
}

/**
 * @brief Implements an erase_if function for different containers according to https://en.cppreference.com/w/cpp/container/map/erase_if.
 * @tparam Container the container type
 * @tparam Pred the type of the unary-predicate
 * @param[in,out] c the container to erase the elements that match the predicate @p pred from
 * @param[in] pred the predicate the to be erased elements must match
 * @return the number of erased elements
 */
template <typename Container, typename Pred>
inline typename Container::size_type erase_if_impl(Container &c, Pred pred) {
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

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the [`std::map`](https://en.cppreference.com/w/cpp/container/map) @p c.
 * @tparam Key the type of the map's key
 * @tparam T the map's value type
 * @tparam Compare the map's comparator type
 * @tparam Allocator the map's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the map
 * @param[in,out] c the map to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the map
 * @return the number of erased elements
 */
template <typename Key, typename T, typename Compare, typename Allocator, typename Pred>
inline typename std::map<Key, T, Compare, Allocator>::size_type erase_if(std::map<Key, T, Compare, Allocator> &c, Pred pred) {
    return erase_if_impl(c, pred);
}

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the [`std::unordered_map`](https://en.cppreference.com/w/cpp/container/unordered_map) @p c.
 * @tparam Key the type of the unordered_map's key
 * @tparam T the unordered_map's value type
 * @tparam Hash the unordered_map's hash function
 * @tparam KeyEqual the unordered_map's function to determine if two keys are equal
 * @tparam Allocator the unordered_map's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the unordered_map
 * @param[in,out] c the unordered_map to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the unordered_map
 * @return the number of erased elements
 */
template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator, typename Pred>
inline typename std::unordered_map<Key, T, Hash, KeyEqual, Allocator>::size_type erase_if(std::unordered_map<Key, T, Hash, KeyEqual, Allocator> &c, Pred pred) {
    return erase_if_impl(c, pred);
}

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the [`std::set`](https://en.cppreference.com/w/cpp/container/set) @p c.
 * @tparam Key the type of the set's key
 * @tparam Compare the set's compare function
 * @tparam Allocator the set's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the set
 * @param[in,out] c the set to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the set
 * @return the number of erased elements
 */
template <typename Key, typename Compare, typename Allocator, typename Pred>
inline typename std::set<Key, Compare, Allocator>::size_type erase_if(std::set<Key, Compare, Allocator> &c, Pred pred) {
    return erase_if_impl(c, pred);
}

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the [`std::unordered_set`](https://en.cppreference.com/w/cpp/container/unordered_set) @p c.
 * @tparam Key the type of the unordered_set's key
 * @tparam Hash the unordered_set's hash function
 * @tparam KeyEqual the unordered_set's function to determine if two keys are equal
 * @tparam Allocator the unordered_set's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the unordered_set
 * @param[in,out] c the unordered_set to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the unordered_set
 * @return the number of erased elements
 */
template <typename Key, typename Hash, typename KeyEqual, typename Allocator, typename Pred>
inline typename std::unordered_set<Key, Hash, KeyEqual, Allocator>::size_type erase_if(std::unordered_set<Key, Hash, KeyEqual, Allocator> &c, Pred pred) {
    return erase_if_impl(c, pred);
}

/**
 * @brief Erases all elements that satisfy the predicate @p pred from the [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) @p vec.
 * @tparam T the value type of the vector
 * @tparam Allocator the vector's allocator type
 * @tparam Pred the type of the predicate used to erase unwanted elements from the vector
 * @param[in, out] vec the vector to erase the elements from
 * @param[in] pred the predicate used to select the elements which will be erased from the vector
 * @return the number of erased elements
 */
template <typename T, typename Allocator, typename Pred>
inline typename std::vector<T, Allocator>::size_type erase_if(std::vector<T, Allocator> &vec, Pred pred) {
    auto it = std::remove_if(vec.begin(), vec.end(), pred);
    auto r = std::distance(it, vec.end());
    vec.erase(it, vec.end());
    return r;
}

/**
 * @brief Check whether the [`std::map`](https://en.cppreference.com/w/cpp/container/map) @p map contains the key @p key.
 * @tparam Key the type of the map's key
 * @tparam T the map's value type
 * @tparam Compare the map's comparator type
 * @tparam Allocator the map's allocator type
 * @param[in,out] map the map which may contain the @p key
 * @param[in] key the key to check
 * @return `true` if the @p key exists in the @p map, otherwise `false` (`[[nodiscard]]`)
 */
template <typename Key, typename T, typename Compare, typename Allocator>
[[nodiscard]] inline bool contains(const std::map<Key, T, Compare, Allocator> &map, const Key &key) {
    return map.count(key) > 0;
}

/**
 * @brief Check whether the [`std::unordered_map`](https://en.cppreference.com/w/cpp/container/unordered_map) @p map contains the key @p key.
 * @tparam Key the type of the unordered_map's key
 * @tparam T the unordered_map's value type
 * @tparam Hash the unordered_map's hash function
 * @tparam KeyEqual the unordered_map's function to determine if two keys are equal
 * @tparam Allocator the unordered_map's allocator type
 * @param[in,out] map the map which may contain the @p key
 * @param[in] key the key to check
 * @return `true` if the @p key exists in the @p map, otherwise `false` (`[[nodiscard]]`)
 */
template <typename Key, typename T, typename Hash, typename KeyEqual, typename Allocator>
[[nodiscard]] inline bool contains(const std::unordered_map<Key, T, Hash, KeyEqual, Allocator> &map, const Key &key) {
    return map.count(key) > 0;
}

/**
 * @brief Check whether the [`std::set`](https://en.cppreference.com/w/cpp/container/set) @p set contains the key @p key.
 * @tparam Key the type of the set's key
 * @tparam Compare the set's compare function
 * @tparam Allocator the set's allocator type
 * @param[in,out] set the set which may contain the @p key
 * @param[in] key the key to check
 * @return `true` if the @p key exists in the @p set, otherwise `false` (`[[nodiscard]]`)
 */
template <typename Key, typename Compare, typename Allocator>
[[nodiscard]] inline bool contains(const std::set<Key, Compare, Allocator> &set, const Key &key) {
    return set.count(key) > 0;
}

/**
 * @brief Check whether the [`std::unordered_set`](https://en.cppreference.com/w/cpp/container/unordered_set) @p set contains the key @p key.
 * @tparam Key the type of the unordered_set's key
 * @tparam Hash the unordered_set's hash function
 * @tparam KeyEqual the unordered_set's function to determine if two keys are equal
 * @tparam Allocator the unordered_set's allocator type
 * @param[in,out] set the set which may contain the @p key
 * @param[in] key the key to check
 * @return `true` if the @p key exists in the @p set, otherwise `false` (`[[nodiscard]]`)
 */
template <typename Key, typename Hash, typename KeyEqual, typename Allocator>
[[nodiscard]] inline bool contains(const std::unordered_set<Key, Hash, KeyEqual, Allocator> &set, const Key &key) {
    return set.count(key) > 0;
}

/**
 * @brief Check whether the [`std::vector`](https://en.cppreference.com/w/cpp/container/vector) @p vec contains the value @p val.
 * @tparam T the value type of the vector
 * @tparam Allocator the vector's allocator type
 * @param[in] vec the vector which may contain the @p val
 * @param[in] val the value to check
 * @return `true` if the @p val exists in the @p vec, otherwise `false` (`[[nodiscard]]`)
 */
template <typename T, typename Allocator>
[[nodiscard]] inline bool contains(const std::vector<T, Allocator> &vec, const T val) {
    return std::find(vec.cbegin(), vec.cend(), val) != vec.cend();
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_UTILITY_HPP_