/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/string_utility.hpp"

#include <algorithm>    // std::min, std::transform
#include <cctype>       // std::tolower, std::toupper
#include <string>       // std::char_traits, std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail {

bool starts_with(const std::string_view str, const std::string_view sv) noexcept {
    return str.substr(0, sv.size()) == sv;
}
bool starts_with(const std::string_view str, const char c) noexcept {
    return !str.empty() && std::char_traits<char>::eq(str.front(), c);
}
bool ends_with(const std::string_view str, const std::string_view sv) noexcept {
    return str.size() >= sv.size() && str.compare(str.size() - sv.size(), std::string_view::npos, sv) == 0;
}
bool ends_with(const std::string_view str, const char c) noexcept {
    return !str.empty() && std::char_traits<char>::eq(str.back(), c);
}
bool contains(const std::string_view str, const std::string_view sv) noexcept {
    return str.find(sv) != std::string_view::npos;
}
bool contains(const std::string_view str, const char c) noexcept {
    return str.find(c) != std::string_view::npos;
}

std::string_view trim_left(const std::string_view str) noexcept {
    const std::string_view::size_type pos = std::min(str.find_first_not_of(' '), str.size());
    return str.substr(pos);
}
std::string_view trim_right(const std::string_view str) noexcept {
    const std::string_view::size_type pos = std::min(str.find_last_not_of(' ') + 1, str.size());
    return str.substr(0, pos);
}
std::string_view trim(const std::string_view str) noexcept {
    return trim_left(trim_right(str));
}

std::string &replace_all(std::string &str, const std::string_view what, const std::string_view with) {
    // prevent endless loop if the "what" string is empty -> nothing to do
    if (what.empty()) {
        return str;
    }
    // replace occurrences of "what" with "with"
    for (std::string::size_type pos = 0; std::string::npos != (pos = str.find(what.data(), pos, what.length())); pos += with.length()) {
        str.replace(pos, what.length(), with.data(), with.length());
    }
    return str;
}

std::string &to_lower_case(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return str;
}

std::string as_lower_case(const std::string_view str) {
    std::string lowercase_str{ str };
    std::transform(str.begin(), str.end(), lowercase_str.begin(), [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowercase_str;
}

std::string &to_upper_case(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return str;
}

std::string as_upper_case(const std::string_view str) {
    std::string uppercase_str{ str };
    std::transform(str.begin(), str.end(), uppercase_str.begin(), [](const unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return uppercase_str;
}

std::vector<std::string_view> split(const std::string_view str, const char delim) {
    std::vector<std::string_view> split_str;

    // if the input str is empty, return an empty vector
    if (str.empty()) {
        return split_str;
    }

    std::string_view::size_type pos = 0;
    std::string_view::size_type next = 0;
    while (next != std::string_view::npos) {
        next = str.find_first_of(delim, pos);
        split_str.emplace_back(next == std::string_view::npos ? str.substr(pos) : str.substr(pos, next - pos));
        pos = next + 1;
    }
    return split_str;
}

}  // namespace plssvm::detail