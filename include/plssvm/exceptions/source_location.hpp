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

#pragma once

#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief The `plssvm::source_location` class represents certain information about the source code, such as file names, line numbers or function names.
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
    [[nodiscard]] static source_location current(
        const char *file_name = __builtin_FILE(),
        const char *function_name = __builtin_FUNCTION(),
        const int line = __builtin_LINE(),
        const int column = 0) noexcept {
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
    [[nodiscard]] std::string_view function_name() const noexcept { return function_name_; }
    /**
     * @brief Returns the function name without additional signature information (i.e. return type and parameters)
     *        or `"unknown"` if no information could be retrieved.
     * @return the function name (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string_view file_name() const noexcept { return file_name_; }
    /**
     * @brief Returns the line number or `0` if no information could be retrieved.
     * @return the line number (`[[nodiscard]]`)
     */
    [[nodiscard]] int line() const noexcept { return line_; }
    /**
     * @brief Returns the column number. Always `0`!
     * @return `0` (`[[nodiscard]]`)
     */
    [[nodiscard]] int column() const noexcept { return column_; }

  private:
    std::string_view function_name_ = "unknown";
    std::string_view file_name_ = "unknown";
    int line_ = 0;
    int column_ = 0;
};

}  // namespace plssvm
