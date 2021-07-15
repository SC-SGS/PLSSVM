/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines universal utility functions.
 */

#pragma once

#include <cstddef>  // std::size_t
#include <tuple>    // std::forward_as_tuple, std::get

namespace plssvm::detail {

/**
 * @brief Type-dependent expression that always evaluates to `false`.
 */
template <typename>
constexpr bool always_false_v = false;

/**
 * @brief Get the @p I-th element of the parameter pack @p ts.
 * @tparam I the index of the element to get
 * @tparam Ts the types inside the parameter pack
 * @param[in] ts the values of the parameter pack
 * @return the @p I-th element of @p ts
 */
template <std::size_t I, class... Ts>
decltype(auto) get(Ts &&...ts) {
    static_assert(I < sizeof...(Ts), "Out-of-bounce access: too few elements in parameter pack!");
    return std::get<I>(std::forward_as_tuple(ts...));
}

}  // namespace plssvm::detail