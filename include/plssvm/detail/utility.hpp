/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines universal utility functions.
 */

#pragma once

#include <cstddef>      // std::size_t
#include <tuple>        // std::forward_as_tuple, std::get
#include <type_traits>  // std::underlying_type_t, std::is_enum_v

namespace plssvm::detail {

/**
 * @brief Type-dependent expression that always evaluates to `false`.
 */
template <typename>
constexpr bool always_false_v = false;

/**
 * @brief Get the @p I-th element of the parameter pack @p ts.
 * @tparam I the index of the element to get
 * @tparam Types the types inside the parameter pack
 * @param[in] args the values of the parameter pack
 * @return the @p I-th element of @p args
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
 * @return the integer value of the underlying type of `Enum`, converted from @p e
 */
template <typename Enum>
[[nodiscard]] constexpr std::underlying_type_t<Enum> to_underlying(const Enum e) noexcept {
    static_assert(std::is_enum_v<Enum>, "e must be an enumeration type!");
    return static_cast<std::underlying_type_t<Enum>>(e);
}

}  // namespace plssvm::detail