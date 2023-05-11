/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines some generic type traits used in the PLSSVM library.
 */

#ifndef PLSSVM_DETAIL_TYPE_TRAITS_HPP_
#define PLSSVM_DETAIL_TYPE_TRAITS_HPP_
#pragma once

#include <array>          // std::array
#include <deque>          // std::deque
#include <forward_list>   // std::forward_list
#include <list>           // std::list
#include <map>            // std::map, std::multimap
#include <set>            // std::set, std::multiset
#include <type_traits>    // std::remove_cv_t, std::remove_reference_t, std::false_type, std::true_type
#include <type_traits>    // std::enable_if_t
#include <unordered_map>  // std::unordered_map, std::unordered_multimap
#include <unordered_set>  // std::unordered_set, std::unordered_multiset
#include <vector>         // std:.vector

namespace plssvm::detail {

/**
 * @brief A shorthand macro for the `std::enable_if_t` type trait.
 */
#define PLSSVM_REQUIRES(...) std::enable_if_t<__VA_ARGS__, bool> = true

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
 * @brief Type trait to check whether @p T is a `std::array`.
 * @tparam T the type to check
 */
template <typename T>
struct is_array : std::false_type {};
/**
 * @copybrief plssvm::detail::is_array
 */
template <typename T, std::size_t I>
struct is_array<std::array<T, I>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_array
 */
template <typename T>
constexpr bool is_array_v = is_array<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::vector`.
 * @tparam T the type to check
 */
template <typename T>
struct is_vector : std::false_type {};
/**
 * @copybrief plssvm::detail::is_vector
 */
template <typename T>
struct is_vector<std::vector<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_vector
 */
template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::deque`.
 * @tparam T the type to check
 */
template <typename T>
struct is_deque : std::false_type {};
/**
 * @copybrief plssvm::detail::is_deque
 */
template <typename T>
struct is_deque<std::deque<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_deque
 */
template <typename T>
constexpr bool is_deque_v = is_deque<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::forward_list`.
 * @tparam T the type to check
 */
template <typename T>
struct is_forward_list : std::false_type {};
/**
 * @copybrief plssvm::detail::is_forward_list
 */
template <typename T>
struct is_forward_list<std::forward_list<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_forward_list
 */
template <typename T>
constexpr bool is_forward_list_v = is_forward_list<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::list`.
 * @tparam T the type to check
 */
template <typename T>
struct is_list : std::false_type {};
/**
 * @copybrief plssvm::detail::is_list
 */
template <typename T>
struct is_list<std::list<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_list
 */
template <typename T>
constexpr bool is_list_v = is_list<T>::value;

/**
 * @brief Type trait to check whether @p T is a sequence container.
 */
template <typename T>
constexpr bool is_sequence_container_v = is_array_v<T> || is_vector_v<T> || is_deque_v<T> || is_forward_list_v<T> || is_list_v<T>;

/**
 * @brief Type trait to check whether @p T is a `std::set`.
 * @tparam T the type to check
 */
template <typename T>
struct is_set : std::false_type {};
/**
 * @copybrief plssvm::detail::is_set
 */
template <typename T>
struct is_set<std::set<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_set
 */
template <typename T>
constexpr bool is_set_v = is_set<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::map`.
 * @tparam T the type to check
 */
template <typename T>
struct is_map : std::false_type {};
/**
 * @copybrief plssvm::detail::is_map
 */
template <typename Key, typename T>
struct is_map<std::map<Key, T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_map
 */
template <typename T>
constexpr bool is_map_v = is_map<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::multiset`.
 * @tparam T the type to check
 */
template <typename T>
struct is_multiset : std::false_type {};
/**
 * @copybrief plssvm::detail::is_multiset
 */
template <typename T>
struct is_multiset<std::multiset<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_multiset
 */
template <typename T>
constexpr bool is_multiset_v = is_multiset<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::multimap`.
 * @tparam T the type to check
 */
template <typename T>
struct is_multimap : std::false_type {};
/**
 * @copybrief plssvm::detail::is_multimap
 */
template <typename Key, typename T>
struct is_multimap<std::multimap<Key, T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_multimap
 */
template <typename T>
constexpr bool is_multimap_v = is_multimap<T>::value;

/**
 * @brief Type trait to check whether @p T is a associative container.
 */
template <typename T>
constexpr bool is_associative_container_v = is_set_v<T> || is_map_v<T> || is_multimap_v<T> || is_multiset_v<T>;

/**
 * @brief Type trait to check whether @p T is a `std::unordered_set`.
 * @tparam T the type to check
 */
template <typename T>
struct is_unordered_set : std::false_type {};
/**
 * @copybrief plssvm::detail::is_unordered_set
 */
template <typename T>
struct is_unordered_set<std::unordered_set<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_unordered_set
 */
template <typename T>
constexpr bool is_unordered_set_v = is_unordered_set<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::unordered_map`.
 * @tparam T the type to check
 */
template <typename T>
struct is_unordered_map : std::false_type {};
/**
 * @copybrief plssvm::detail::is_unordered_map
 */
template <typename Key, typename T>
struct is_unordered_map<std::unordered_map<Key, T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_unordered_map
 */
template <typename T>
constexpr bool is_unordered_map_v = is_unordered_map<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::unordered_multiset`.
 * @tparam T the type to check
 */
template <typename T>
struct is_unordered_multiset : std::false_type {};
/**
 * @copybrief plssvm::detail::is_unordered_multiset
 */
template <typename T>
struct is_unordered_multiset<std::unordered_multiset<T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_unordered_multiset
 */
template <typename T>
constexpr bool is_unordered_multiset_v = is_unordered_multiset<T>::value;

/**
 * @brief Type trait to check whether @p T is a `std::unordered_multimap`.
 * @tparam T the type to check
 */
template <typename T>
struct is_unordered_multimap : std::false_type {};
/**
 * @copybrief plssvm::detail::is_unordered_multimap
 */
template <typename Key, typename T>
struct is_unordered_multimap<std::unordered_multimap<Key, T>> : std::true_type {};
/**
 * @copybrief plssvm::detail::is_unordered_multimap
 */
template <typename T>
constexpr bool is_unordered_multimap_v = is_unordered_multimap<T>::value;

/**
 * @brief Type trait to check whether @p T is a unordered associative container.
 */
template <typename T>
constexpr bool is_unordered_associative_container_v = is_unordered_set_v<T> || is_unordered_map_v<T> || is_unordered_multimap_v<T> || is_unordered_multiset_v<T>;

/**
 * @brief Type trait to check whether @p T is a container.
 */
template <typename T>
constexpr bool is_container_v = is_sequence_container_v<T> || is_associative_container_v<T> || is_unordered_associative_container_v<T>;

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_TYPE_TRAITS_HPP_