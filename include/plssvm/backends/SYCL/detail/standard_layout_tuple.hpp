/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implementation of a basic and minimalistic tuple class which is standard-layout conform.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DETAIL_STANDARD_LAYOUT_TUPLE_HPP_
#define PLSSVM_BACKENDS_SYCL_DETAIL_STANDARD_LAYOUT_TUPLE_HPP_

#include "plssvm/constants.hpp"  // plssvm::real_type

#include <cstddef>      // std::size_t
#include <type_traits>  // std::is_standard_layout
#include <utility>      // std::forward

namespace plssvm::sycl::detail {

/*
 * Empty base implementation.
 */
template <typename...>
struct standard_layout_tuple;

/**
 * @brief Save the value of type @p T as scalar and the remaining values of type @p Rest recursively in another standard layout tuple.
 * @tparam T the type of the value to save in this tuple
 * @tparam Rest the remaining types saved in a recursive tuple
 */
template <typename T, typename... Rest>
struct standard_layout_tuple<T, Rest...> {
    /// The stored value.
    T value;
    /// The remaining values stored in their own tuple.
    standard_layout_tuple<Rest...> remaining;
};

/**
 * @brief Special case for an empty tuple (recursion termination criterion).
 */
template <>
struct standard_layout_tuple<> { };

namespace impl {

/**
 * @brief Recursively traverse (at compile time) the tuple @p t and retrieve the value at position @p I.
 * @tparam I the index of the tuple value to get
 */
template <std::size_t I>
struct get_impl {
    /**
     * @brief Recursively traverse (at compile time) the tuple @p t and retrieve the value at position @p I.
     * @tparam Types the types in the tuple
     * @param[in] t the tuple to traverse
     * @return the requested value (`[[nodiscard]]`)
     */
    template <typename... Types>
    [[nodiscard]] constexpr static auto get(const standard_layout_tuple<Types...> &t) {
        return get_impl<I - 1>::get(t.remaining);
    }
};

/**
 * @brief Special case to retrieve the currently held value (recursion termination criterion).
 */
template <>
struct get_impl<0> {
    /**
     * @brief Get the held value from @p t.
     * @tparam Types the types in the tuple
     * @param[in] t the tuple to get the value from
     * @return the requested value (`[[nodiscard]]`)
     */
    template <typename... Types>
    [[nodiscard]] constexpr static auto get(const standard_layout_tuple<Types...> &t) {
        return t.value;
    }
};

}  // namespace impl

/**
 * @brief Get the value at position @p I of the tuple @p t holding the @p Types.
 * @tparam I the position of the element in the tuple to get
 * @tparam Types the types stored in the tuple
 * @param[in] t the tuple
 * @return the value of the tuple @p t at position @p I (`[[nodiscard]]`)
 */
template <std::size_t I, typename... Types>
[[nodiscard]] inline constexpr auto get(const standard_layout_tuple<Types...> &t) {
    static_assert(I < sizeof...(Types), "Invalid standard_layout_tuple index!");
    return impl::get_impl<I>::get(t);
}

/**
 * @brief Special case: return an empty tuple if no values have bee provided.
 * @return an empty tuple (`[[nodiscard]]`)
 */
[[nodiscard]] inline constexpr standard_layout_tuple<> make_standard_layout_tuple() {
    return standard_layout_tuple<>{};
}

/**
 * @brief Create a new tuple storing the values @p arg and @p remaining.
 * @tparam T the type of the first value
 * @tparam Rest the types of the remaining values (if any)
 * @param[in,out] arg the first value
 * @param[in,out] remaining the remaining values (if any)
 * @return the constructed tuple (`[[nodiscard]]`)
 */
template <typename T, typename... Rest>
[[nodiscard]] inline constexpr standard_layout_tuple<T, Rest...> make_standard_layout_tuple(T &&arg, Rest &&...remaining) {
    return standard_layout_tuple<T, Rest...>{ std::forward<T>(arg), make_standard_layout_tuple(std::forward<Rest>(remaining)...) };
}

// sanity checks: be sure that the important use cases are indeed standard layout types!
static_assert(std::is_standard_layout_v<standard_layout_tuple<>>, "standard_layout_tuple<> has no standard layout!");
static_assert(std::is_standard_layout_v<standard_layout_tuple<int, real_type, real_type>>, "standard_layout_tuple<int, real_type, real_type> has no standard layout!");
static_assert(std::is_standard_layout_v<standard_layout_tuple<real_type>>, "standard_layout_tuple<real_type> has no standard layout!");

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_DETAIL_STANDARD_LAYOUT_TUPLE_HPP_
