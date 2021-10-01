/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright
*
* @brief Implements a conversion function from a string to an arithmetic type.
*/

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/utility.hpp"               // plssvm::detail::always_false_v

#include "fast_float/fast_float.h"  // fast_float::from_chars_result, fast_float::from_chars (floating point types)
#include "fmt/core.h"               // fmt::format

#include <charconv>      // std::from_chars_result, std::from_chars (integral types)
#include <stdexcept>     // std::runtime_error
#include <system_error>  // std:errc
#include <type_traits>   // std::is_floating_point_v, std::is_integral_v

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