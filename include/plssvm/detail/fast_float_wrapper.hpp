/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Wrapper around fast_float to prevent UB when using fast-math (which is always disabled in the wrapper library).
 */

#ifndef PLSSVM_DETAIL_FAST_FLOAT_WRAPPER_HPP_
#define PLSSVM_DETAIL_FAST_FLOAT_WRAPPER_HPP_
#pragma once

#include <string_view>   // std::string_view
#include <system_error>  // std:errc
#include <utility>       // std::pair

namespace plssvm::detail {

/**
 * @brief Converts the string @p str to a floating point value of type @p T.
 * @details If @p T is a `long double` [`std::stold`](https://en.cppreference.com/w/cpp/string/basic_string/stof) is used since fast_float doesn't support long double,
 *          otherwise [`float_fast::from_chars`](https://github.com/fastfloat/fast_float) is used.
 * @tparam T the type to convert the value of @p str to, must be a floating point type
 * @param[in] str the string to convert
 * @return the value of type @p T denoted by @p str and the potential error code if the @p str couldn't be converted to the type @p T (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::pair<T, std::errc> convert_to_floating_point(std::string_view str);

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_FAST_FLOAT_WRAPPER_HPP_
