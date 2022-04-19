/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a conversion function from a string to an arithmetic type.
 */

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::trim_left
#include "plssvm/detail/utility.hpp"               // plssvm::detail::always_false_v

#include "fast_float/fast_float.h"  // fast_float::from_chars_result, fast_float::from_chars (floating point types)
#include "fmt/core.h"               // fmt::format

#include <charconv>      // std::from_chars_result, std::from_chars (integral types)
#include <stdexcept>     // std::runtime_error
#include <string_view>   // std::string_view
#include <system_error>  // std:errc
#include <type_traits>   // std::is_floating_point_v, std::is_integral_v

namespace plssvm::detail {

/**
 * @brief Converts the string @p str to a value of type @p T.
 * @details If @p T is an integral type [`std::from_chars`](https://en.cppreference.com/w/cpp/utility/from_chars) is used,
 *          if @p T is a floating point type [`float_fast::from_chars`](https://github.com/fastfloat/fast_float) is used
 *          and if @p T is neither of both, an exception is thrown.
 * @tparam T the type to convert the value of @p str to, must be an arithmetic type
 * @tparam Exception the exception type to throw in case that @p str can't be converted to a value of @p T
 *         (default: [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)).
 * @param[in] str the string to convert
 * @throws Exception if @p str can't be converted to a value of type @p T
 * @return the value of type @p T denoted by @p str (`[[nodiscard]]`)
 */
template <typename T, typename Exception = std::runtime_error>
[[nodiscard]] inline T convert_to(std::string_view str) {
    // select conversion function depending on the provided type
    const auto convert_from_chars = [](const std::string_view sv, auto &val) {
        if constexpr (std::is_floating_point_v<T>) {
            // convert the string to a floating point value
            return fast_float::from_chars(sv.data(), sv.data() + sv.size(), val);
        } else if constexpr (std::is_integral_v<T>) {
            // convert the string to an integral value
            return std::from_chars(sv.data(), sv.data() + sv.size(), val);
        } else {
            // can't convert the string to a non-arithmetic type
            static_assert(always_false_v<T>, "Can only convert arithmetic types!");
        }
    };

    // remove leading whitespaces
    str = trim_left(str);

    // convert string to value fo type T
    T val;
    auto res = convert_from_chars(str, val);
    if (res.ec != std::errc{}) {
        throw Exception{ fmt::format("Can't convert '{}' to a value of type {}!", str, plssvm::detail::arithmetic_type_name<T>()) };
    }
    return val;
}

/**
 * @brief Extract the first integer from the given string @p str and converts it to @p T.
 * @tparam T the type to convert the first integer to
 * @param[in] str the string to check
 * @throws std::runtime_error if @p str doesn't contain an integer
 * @return the converted integer of type @p T (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T extract_first_integer_from_string(std::string_view str) {
    const std::string_view::size_type n = str.find_first_of("0123456789");
    if (n != std::string_view::npos) {
        const std::string_view::size_type m = str.find_first_not_of("0123456789", n);
        return convert_to<T>(str.substr(n, m != std::string_view::npos ? m - n : m));
    }
    throw std::runtime_error{ fmt::format("String {} doesn't contain any integer!", str) };
}

}  // namespace plssvm::detail