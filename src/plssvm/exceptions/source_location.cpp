/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/exceptions/source_location.hpp"

#include <string_view>  // std::string_view

namespace plssvm {

source_location source_location::current(const char *file_name, const char *function_name, const int line, const int column) noexcept {
    source_location loc;

    loc.file_name_ = file_name;
    loc.function_name_ = function_name;
    loc.line_ = line;
    loc.column_ = column;

    return loc;
}

std::string_view source_location::function_name() const noexcept { return function_name_; }
std::string_view source_location::file_name() const noexcept { return file_name_; }
int source_location::line() const noexcept { return line_; }
int source_location::column() const noexcept { return column_; }

}  // namespace plssvm