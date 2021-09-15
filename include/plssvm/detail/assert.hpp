/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements a custom assert macro.
 */

#pragma once

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include "fmt/color.h"  // fmt::emphasis, fmt::fg, fmt::color
#include "fmt/core.h"   // fmt::print, fmt::format

#include <cstdio>       // stderr
#include <cstdlib>      // std::abort
#include <string_view>  // std::string_view

namespace plssvm::detail {

/**
 * @brief Function called by the `PLSSVM_ASSERT` macro. Checks the assertion condition and prints and aborts the program if the condition evaluates to `false`.
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
        fmt::print(stderr,
                   "Assertion '{}' failed!\n"
                   "  in file      {}\n"
                   "  in function  {}\n"
                   "  @ line       {}\n\n"
                   "{}\n",
                   fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::green), cond_str),
                   loc.file_name(),
                   loc.function_name(),
                   loc.line(),
                   fmt::format(fmt::emphasis::bold | fmt::fg(fmt::color::red), msg, std::forward<Args>(args)...));

        // abort further execution
        std::abort();
    }
}

/**
 * @def PLSSVM_ASSERT
 * @brief Defines the `PLSSVM_ASSERT` macro if PLSSVM_ENABLE_ASSERTS is defined or if `NDEBUG` is **not** defined (in DEBUG mode).
 */
#if defined(PLSSVM_ENABLE_ASSERTS) || !defined(NDEBUG)
    #define PLSSVM_ASSERT(cond, msg, ...) plssvm::detail::check_assertion(cond, #cond, plssvm::source_location::current(), msg, ##__VA_ARGS__)
#else
    #define PLSSVM_ASSERT(cond, msg, ...)
#endif

}  // namespace plssvm::detail