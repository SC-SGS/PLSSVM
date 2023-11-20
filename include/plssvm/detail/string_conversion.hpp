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

#ifndef PLSSVM_DETAIL_STRING_CONVERSION_HPP_
#define PLSSVM_DETAIL_STRING_CONVERSION_HPP_
#pragma once

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::{trim, trim_left, as_lower_case}
#include "plssvm/detail/type_traits.hpp"           // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t

#include "fast_float/fast_float.h"  // fast_float::from_chars_result, fast_float::from_chars (floating point types)
#include "fmt/core.h"               // fmt::format

#include <charconv>      // std::from_chars_result, std::from_chars (integral types)
#include <stdexcept>     // std::runtime_error
#include <string>        // std::string, std::stold
#include <string_view>   // std::string_view
#include <system_error>  // std:errc
#include <type_traits>   // std::is_arithmetic_v, std::is_same_v, std::is_floating_point_v, std::is_integral_v
#include <vector>        // std::vector

namespace plssvm::detail {

/**
 * @brief Converts the string @p str to a value of type @p T.
 * @details If @p T is an integral type [`std::from_chars`](https://en.cppreference.com/w/cpp/utility/from_chars) is used,
 *          if @p T is a floating point type [`float_fast::from_chars`](https://github.com/fastfloat/fast_float) is used,
 *          if @p T is a `std::string` the normal `std::string` constructor is used,
 *          and if @p T is neither of the three, an exception is thrown.
 * @tparam T the type to convert the value of @p str to, must be an arithmetic type or a `std::string`
 * @tparam Exception the exception type to throw in case that @p str can't be converted to a value of @p T
 *         (default: [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)).
 * @param[in] str the string to convert
 * @throws Exception if @p str can't be converted to a value of type @p T
 * @return the value of type @p T denoted by @p str (`[[nodiscard]]`)
 */
template <typename T, typename Exception = std::runtime_error, PLSSVM_REQUIRES((std::is_arithmetic_v<T> || std::is_same_v<remove_cvref_t<T>, std::string>) )>
[[nodiscard]] inline T convert_to(const std::string_view str) {
    if constexpr (std::is_same_v<remove_cvref_t<T>, std::string>) {
        // convert string_view to string
        return std::string{ trim(str) };
    } else if constexpr (std::is_same_v<remove_cvref_t<T>, bool>) {
        const std::string lower_case_str = as_lower_case(trim(str));
        // the string true
        if (lower_case_str == "true") {
            return true;
        }
        // the string false
        if (lower_case_str == "false") {
            return false;
        }
        // convert a number to its "long long" value and convert it to a bool: 0 -> false, otherwise true
        return static_cast<bool>(convert_to<long long, Exception>(str));
    } else if constexpr (std::is_same_v<remove_cvref_t<T>, char>) {
        const std::string_view trimmed = trim(str);
        if (trimmed.size() != 1) {
            throw Exception{ fmt::format("Can't convert '{}' to a value of type char!", str) };
        }
        return trimmed.front();
    } else if constexpr (std::is_same_v<remove_cvref_t<T>, long double>) {
        return std::stold(std::string{ str });
    } else {
        // select conversion function depending on the provided type
        const auto convert_from_chars = [](const std::string_view sv, auto &val) {
            if constexpr (std::is_floating_point_v<T>) {
                // convert the string to a floating point value
                return fast_float::from_chars(sv.data(), sv.data() + sv.size(), val);
            } else {
                // convert the string to an integral value
                return std::from_chars(sv.data(), sv.data() + sv.size(), val);
            }
            // unreachable, needed to silence a nvcc warning
            // see also: https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
            return std::conditional_t<std::is_floating_point_v<T>, fast_float::from_chars_result, std::from_chars_result>{};
        };

        // remove leading whitespaces
        const std::string_view trimmed_str = trim_left(str);

        // convert string to value fo type T
        T val;
        auto res = convert_from_chars(trimmed_str, val);
        if (res.ec != std::errc{}) {
            throw Exception{ fmt::format("Can't convert '{}' to a value of type {}!", str, arithmetic_type_name<T>()) };
        }
        return val;
    }
    // unreachable, needed to silence a nvcc warning
    // see also: https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
    return T{};
}

/**
 * @brief Extract the first integer from the given string @p str and converts it to @p T ignoring a potential sign.
 * @tparam T the type to convert the first integer to
 * @param[in] str the string to check
 * @throws std::runtime_error if @p str doesn't contain an integer
 * @return the converted first integer of type @p T (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T extract_first_integer_from_string(std::string_view str) {
    const std::string_view::size_type n = str.find_first_of("0123456789");
    if (n != std::string_view::npos) {
        const std::string_view::size_type m = str.find_first_not_of("0123456789", n);
        return convert_to<T>(str.substr(n, m != std::string_view::npos ? m - n : m));
    }
    throw std::runtime_error{ fmt::format("String \"{}\" doesn't contain any integer!", str) };
}

/**
 * @brief Split the string @p str at the positions with delimiter @p delim and return the sub-strings **converted** to the type @p T.
 * @tparam T the type to convert the value of @p str to, must be an arithmetic type or a `std::string`
 * @param[in] str the string to split
 * @param[in] delim the split delimiter
 * @return the split sub-strings **converted** to the type @p T (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> split_as(const std::string_view str, const char delim = ' ') {
    std::vector<T> split_str;

    // if the input str is empty, return an empty vector
    if (str.empty()) {
        return split_str;
    }

    std::string_view::size_type pos = 0;
    std::string_view::size_type next = 0;
    while (next != std::string_view::npos) {
        next = str.find_first_of(delim, pos);
        split_str.emplace_back(convert_to<T>(next == std::string_view::npos ? str.substr(pos) : str.substr(pos, next - pos)));
        pos = next + 1;
    }
    return split_str;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_STRING_CONVERSION_HPP_