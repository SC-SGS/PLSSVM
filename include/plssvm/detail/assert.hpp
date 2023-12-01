/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom assert macro `PLSSVM_ASSERT`.
 */

#ifndef PLSSVM_DETAIL_ASSERT_HPP_
#define PLSSVM_DETAIL_ASSERT_HPP_
#pragma once

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include "fmt/color.h"  // fmt::emphasis, fmt::fg, fmt::color
#include "fmt/core.h"   // fmt::format

#include <cstdlib>      // std::abort
#include <iostream>     // std::cerr, std::endl
#include <string_view>  // std::string_view
#include <utility>      // std::forward

namespace plssvm::detail {

/**
 * @brief Function called by the `PLSSVM_ASSERT` macro. Checks the assertion condition. If the condition evaluates to `false`,
 *        prints the assertion condition together with additional information (e.g., plssvm::source_location information) and aborts the program.
 * @tparam Args the placeholder types
 * @param[in] cond the assertion condition, aborts the program if evaluated to `false`
 * @param[in] cond_str the assertion condition as string
 * @param[in] loc the source location where the assertion appeared
 * @param[in] msg the custom assertion message
 * @param[in] args the placeholder values
 */
template <typename... Args>
inline void check_assertion(const bool cond, const std::string_view cond_str, const source_location &loc, const std::string_view msg, Args &&...args) {
    // check if assertion holds
    if (!cond) {
        // print assertion error message
        std::cerr << fmt::format(
            "Assertion '{}' failed!\n"
            "  in file      {}\n"
            "  in function  {}\n"
            "  @ line       {}\n\n"
            "{}\n",
            fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::green), "{}", cond_str),
            loc.file_name(),
            loc.function_name(),
            loc.line(),
            fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::red), msg, std::forward<Args>(args)...))
                  << std::endl;

        // abort further execution
        std::abort();
    }
}

/**
 * @def PLSSVM_ASSERT_ENABLED
 * @brief Defines the `PLSSVM_ASSERT_ENABLED` if `PLSSVM_ENABLE_ASSERTS` is defined and `NDEBUG` is **not** defined (in DEBUG mode).
 */
#if defined(PLSSVM_ENABLE_ASSERTS) || !defined(NDEBUG)
    #define PLSSVM_ASSERT_ENABLED 1
#endif

/**
 * @def PLSSVM_ASSERT
 * @brief Defines the `PLSSVM_ASSERT` macro if `PLSSVM_ASSERT_ENABLED` is defined.
 */
#if defined(PLSSVM_ASSERT_ENABLED)
    #define PLSSVM_ASSERT(cond, msg, ...) plssvm::detail::check_assertion((cond), (#cond), plssvm::source_location::current(), (msg), ##__VA_ARGS__)
#else
    #define PLSSVM_ASSERT(cond, msg, ...)
#endif

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_ASSERT_HPP_