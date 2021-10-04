/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functions for string manipulation and conversion.
 */

#pragma once

#include <algorithm>    // std::min, std::transform
#include <cctype>       // std::tolower
#include <string>       // std::char_traits, std::string
#include <string_view>  // std::string_view

namespace plssvm::detail {

/**
 * @brief Checks if the string @p str starts with the prefix @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to match against the start of @p str
 * @return `true` if @p str starts with the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool starts_with(const std::string_view str, const std::string_view sv) noexcept {
    return str.substr(0, sv.size()) == sv;
}
/**
 * @brief Checks if the string @p str starts with the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to match against the first character of @p str
 * @return `true` if @p str starts with the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool starts_with(const std::string_view str, const char c) noexcept {
    return !str.empty() && std::char_traits<char>::eq(str.front(), c);
}
/**
 * @brief Checks if the string @p str ends with the suffix @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to match against the end of @p str
 * @return `true` if @p str ends with the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool ends_with(const std::string_view str, const std::string_view sv) noexcept {
    return str.size() >= sv.size() && str.compare(str.size() - sv.size(), std::string_view::npos, sv) == 0;
}
/**
 * @brief Checks if the string @p str ends with the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to match against the last character of @p str
 * @return `true` if @p str ends with the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool ends_with(const std::string_view str, const char c) noexcept {
    return !str.empty() && std::char_traits<char>::eq(str.back(), c);
}
/**
 * @brief Checks if the string @p str contains the string @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to find
 * @return `true` if @p str contains the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool contains(const std::string_view str, const std::string_view sv) noexcept {
    return str.find(sv) != std::string_view::npos;
}
/**
 * @brief Checks if the string @p str contains the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to find
 * @return `true` if @p str contains the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool contains(const std::string_view str, const char c) noexcept {
    return str.find(c) != std::string_view::npos;
}

/**
 * @brief Returns a new [`std::string_view`](https://en.cppreference.com/w/cpp/string/basic_string_view) equal to @p str where all leding whitespaces are removed.
 * @param[in] str the string to remove the leading whitespaces
 * @return the string @p str without leading whitespace (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::string_view trim_left(const std::string_view str) noexcept {
    std::string_view::size_type pos = std::min(str.find_first_not_of(' '), str.size());
    return str.substr(pos);
}

/**
 * @brief Replaces all occurrences of @p what with @p with in the string @p str.
 * @param[in,out] str the string to replace the values
 * @param[in] what the string to replace
 * @param[in] with the string to replace with
 */
inline void replace_all(std::string &str, const std::string_view what, const std::string_view with) {
    for (std::string::size_type pos = 0; std::string::npos != (pos = str.find(what.data(), pos, what.length())); pos += with.length()) {
        str.replace(pos, what.length(), with.data(), with.length());
    }
}

/**
 * @brief Convert the string @p str to its all lower case representation.
 * @param[in,out] str the string to transform
 * @return the transformed string
 */
inline std::string &to_lower_case(std::string &str) {
    std::transform(str.begin(), str.end(), str.begin(), [](const char c) { return std::tolower(c); });
    return str;
}

/**
 * @brief Return a new string with the same content as @p str but all lower case.
 * @details In contrast to `std::string& to_lower_case(std::string&)` this function does not change the input string @p str.
 * @param[in] str the string to use in the transformation
 * @return the transformed string (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::string as_lower_case(const std::string_view str) {
    std::string lowercase_str{ str };
    std::transform(str.begin(), str.end(), lowercase_str.begin(), [](const char c) { return std::tolower(c); });
    return lowercase_str;
}

}  // namespace plssvm::detail