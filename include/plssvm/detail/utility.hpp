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

#include "plssvm/default_value.hpp"       // plssvm::default_value
#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size
#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES, plssvm::detail::always_false_v

#include <algorithm>    // std::remove_if, std::find
#include <cstddef>      // std::size_t
#include <iterator>     // std::distance
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::forward_as_tuple, std::get
#include <type_traits>  // std::underlying_type_t, std::is_enum_v

/**
 * @brief Helper function for an extra round of macro expansion inside the PLSSVM_IS_DEFINED macro.
 */
#define PLSSVM_IS_DEFINED_HELPER(x) #x
/**
 * @brief Evaluates to `true` if the preprocessor macro @p x is defined, otherwise `false`.
 * @details Based on: https://stackoverflow.com/questions/18048039/c-constexpr-function-to-test-preprocessor-macros
 */
#define PLSSVM_IS_DEFINED(x) (std::string_view{ #x } != std::string_view{ PLSSVM_IS_DEFINED_HELPER(x) })

#if defined(_WIN32)
#if defined(PLSSVM_COMPILE_BASE_LIBRARY)
    #define PLSSVM_EXTERN extern "C" __declspec(dllexport)
#else
    #define PLSSVM_EXTERN extern "C" __declspec(dllimport)
#endif
#else
    #define PLSSVM_EXTERN extern
#endif

namespace plssvm::detail {

/**
 * @brief Invokes undefined behavior. Used to mark code paths that may never be reachable.
 * @details See: C++23 [`std::unreachable`](https://en.cppreference.com/w/cpp/utility/unreachable)
 */
[[noreturn]] inline void unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(__GNUC__)  // GCC, Clang, ICC
    __builtin_unreachable();
#elif defined(_MSC_VER)  // MSVC
    __assume(false);
#endif
}

/**
 * @brief Get the @p I-th element of the parameter pack @p args at compile-time.
 * @tparam I the index of the element to get
 * @tparam Types the types inside the parameter pack
 * @param[in] args the values of the parameter pack
 * @return the @p I-th element of @p args (`[[nodiscard]]`)
 */
template <std::size_t I, typename... Types>
[[nodiscard]] constexpr decltype(auto) get(Types &&...args) noexcept {
    static_assert(I < sizeof...(Types), "Out-of-bounce access!: too few elements in parameter pack");
    return std::get<I>(std::forward_as_tuple(std::forward<Types>(args)...));
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
template <typename Container, typename Pred, PLSSVM_REQUIRES(is_container_v<Container>)>
inline typename Container::size_type erase_if(Container &c, Pred pred) {
    if constexpr (is_vector_v<Container>) {
        // use optimized version for std::vector
        auto iter = std::remove_if(c.begin(), c.end(), pred);
        auto dist = std::distance(iter, c.end());
        c.erase(iter, c.end());
        return dist;
    } else {
        // generic version otherwise
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
}

/**
 * @brief Check whether the Container @p c contains the value @p val.
 * @tparam Container the container type
 * @tparam T the type of the value to check
 * @param[in] c the container to check if it contains the value @p val
 * @param[in] val the value to check
 * @return `true` if the @p val exists in the container @p c, otherwise `false` (`[[nodiscard]]`)
 */
template <typename Container, typename T, PLSSVM_REQUIRES(is_container_v<Container>)>
[[nodiscard]] inline bool contains(const Container &c, const T &val) {
    if constexpr (is_sequence_container_v<Container>) {
        // use std::find for sequence containers
        return std::find(c.cbegin(), c.cend(), val) != c.cend();
    } else {
        // use count otherwise
        return c.count(val) > 0;
    }
}

/**
 * @brief Return the current date time in the format "YYYY-MM-DD hh:mm:ss".
 * @return the current date time (`[[nodiscard]]`)
 */
[[nodiscard]] std::string current_date_time();

/**
 * @brief Returns the available total system memory.
 * @return the total system memory in bytes (`[[nodiscard]]`)
 */
[[nodiscard]] memory_size get_system_memory();

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_UTILITY_HPP_