/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/string_conversion.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::trim_left
#include "plssvm/detail/type_traits.hpp"     // plssvm::remove_cvref_t

#include "fast_float/fast_float.h"  // fast_float::from_chars_result, fast_float::from_chars

#include <string>        // std::stold, std::string
#include <string_view>   // std::string_view
#include <system_error>  // std::errc
#include <type_traits>   // std::is_same_v, std::is_floating_point_v
#include <utility>       // std::pair, std::make_pair

namespace plssvm::detail {

template <typename T>
std::pair<T, std::errc> convert_to_floating_point(const std::string_view str) {
    static_assert(std::is_floating_point_v<T>, "'convert_to_floating_point' may only be called with template types 'T' which are floating points!");
    if constexpr (std::is_same_v<remove_cvref_t<T>, long double>) {
        return std::make_pair(std::stold(std::string{ str }), std::errc{});
    } else {
        // remove leading whitespaces
        const std::string_view trimmed_str = trim_left(str);

        // convert string to value fo type T
        T val;
        const fast_float::from_chars_result res = fast_float::from_chars(trimmed_str.data(), trimmed_str.data() + trimmed_str.size(), val);
        return std::make_pair(val, res.ec);
    }
}

template std::pair<float, std::errc> convert_to_floating_point(const std::string_view);
template std::pair<double, std::errc> convert_to_floating_point(const std::string_view);
template std::pair<long double, std::errc> convert_to_floating_point(const std::string_view);

}  // namespace plssvm::detail
