/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom [`std::source_location`](https://en.cppreference.com/w/cpp/utility/source_location) implementation for C++17.
 */

#ifndef PLSSVM_EXCEPTIONS_SOURCE_LOCATION_HPP_
#define PLSSVM_EXCEPTIONS_SOURCE_LOCATION_HPP_
#pragma once

#include <cstdint>      // std::uint_least32_t
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief The plssvm::source_location class represents certain information about the source code, such as file names, line numbers, or function names.
 * @details Based on [`std::source_location`](https://en.cppreference.com/w/cpp/utility/source_location).
 */
class source_location {
  public:
    /**
     * @brief Construct new source location information about the current call side.
     * @param[in] file_name the file name including its absolute path, as given by `__builtin_FILE()`
     * @param[in] function_name the function name (without return type and parameters), as given by `__builtin_FUNCTION()`
     * @param[in] line the line number, as given by `__builtin_LINE()`
     * @param[in] column the column number, always `0`
     * @return the source location object holding the information about the current call side (`[[nodiscard]]`)
     */
    [[nodiscard]] static constexpr source_location current(
        const char *file_name = __builtin_FILE(),
        const char *function_name = __builtin_FUNCTION(),
        int line = __builtin_LINE(),
        int column = 0) noexcept {
        source_location loc;

        loc.file_name_ = file_name;
        loc.function_name_ = function_name;
        loc.line_ = line;
        loc.column_ = column;

        return loc;
    }

    /**
     * @brief Returns the absolute path name of the file or `"unknown"` if no information could be retrieved.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr std::string_view function_name() const noexcept { return function_name_; }
    /**
     * @brief Returns the function name without additional signature information (i.e. return type and parameters)
     *        or `"unknown"` if no information could be retrieved.
     * @return the function name (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr std::string_view file_name() const noexcept { return file_name_; }
    /**
     * @brief Returns the line number or `0` if no information could be retrieved.
     * @return the line number (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr std::uint_least32_t line() const noexcept { return line_; }
    /**
     * @brief Returns the column number.
     * @attention Always `0`!
     * @return `0` (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr std::uint_least32_t column() const noexcept { return column_; }

  private:
    /// The line number as retrieved by `__builtin_LINE()`.
    std::uint_least32_t line_{ 0 };
    /// The column number (always `0`).
    std::uint_least32_t column_{ 0 };
    /// The file name as retrieved by `__builtin_FILE()`.
    const char *file_name_{ "unknown" };
    /// The function name as retrieved by `__builtin_FUNCTION()`.
    const char *function_name_{ "unknown" };
};

}  // namespace plssvm

#endif  // PLSSVM_EXCEPTIONS_SOURCE_LOCATION_HPP_