/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functions for string manipulation and querying.
 */

#ifndef PLSSVM_DETAIL_STRING_UTILITY_HPP_
#define PLSSVM_DETAIL_STRING_UTILITY_HPP_
#pragma once

#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail {

/**
 * @brief Checks if the string @p str starts with the prefix @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to match against the start of @p str
 * @return `true` if @p str starts with the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool starts_with(std::string_view str, std::string_view sv) noexcept;
/**
 * @brief Checks if the string @p str starts with the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to match against the first character of @p str
 * @return `true` if @p str starts with the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool starts_with(std::string_view str, char c) noexcept;
/**
 * @brief Checks if the string @p str ends with the suffix @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to match against the end of @p str
 * @return `true` if @p str ends with the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool ends_with(std::string_view str, std::string_view sv) noexcept;
/**
 * @brief Checks if the string @p str ends with the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to match against the last character of @p str
 * @return `true` if @p str ends with the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool ends_with(std::string_view str, char c) noexcept;
/**
 * @brief Checks if the string @p str contains the string @p sv.
 * @param[in] str the string to check
 * @param[in] sv the string to find
 * @return `true` if @p str contains the string @p sv, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool contains(std::string_view str, std::string_view sv) noexcept;
/**
 * @brief Checks if the string @p str contains the character @p c.
 * @param[in] str the string to check
 * @param[in] c the character to find
 * @return `true` if @p str contains the character @p c, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool contains(std::string_view str, char c) noexcept;

/**
 * @brief Returns a new [`std::string_view`](https://en.cppreference.com/w/cpp/string/basic_string_view) equal to @p str where all leading whitespaces are removed.
 * @param[in] str the string to remove the leading whitespaces
 * @return the string @p str without leading whitespace (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view trim_left(std::string_view str) noexcept;
/**
 * @brief Returns a new [`std::string_view`](https://en.cppreference.com/w/cpp/string/basic_string_view) equal to @p str where all trailing whitespaces are removed.
 * @param[in] str the string to remove the trailing whitespaces
 * @return the string @p str without trailing whitespace (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view trim_right(std::string_view str) noexcept;
/**
 * @brief Returns a new [`std::string_view`](https://en.cppreference.com/w/cpp/string/basic_string_view) equal to @p str where all leading and trailing whitespaces are removed.
 * @param[in] str the string to remove the leading and trailing whitespaces
 * @return the string @p str without leading and trailing whitespace (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view trim(std::string_view str) noexcept;

/**
 * @brief Replaces all occurrences of @p what with @p with in the string @p str.
 * @param[in,out] str the string to replace the values
 * @param[in] what the string to replace
 * @param[in] with the string to replace with
 * @return the replaced string
 */
std::string &replace_all(std::string &str, std::string_view what, std::string_view with);

/**
 * @brief Convert the string @p str to its all lower case representation.
 * @param[in,out] str the string to transform
 * @return the transformed string
 */
std::string &to_lower_case(std::string &str);

/**
 * @brief Return a new string with the same content as @p str but all lower case.
 * @details In contrast to to_lower_case(std::string&) this function does not change the input string @p str.
 * @param[in] str the string to use in the transformation
 * @return the transformed string (`[[nodiscard]]`)
 */
[[nodiscard]] std::string as_lower_case(std::string_view str);

/**
 * @brief Convert the string @p str to its all upper case representation.
 * @param[in,out] str the string to transform
 * @return the transformed string
 */
std::string &to_upper_case(std::string &str);

/**
 * @brief Return a new string with the same content as @p str but all upper case.
 * @details In contrast to to_upper_case(std::string&) this function does not change the input string @p str.
 * @param[in] str the string to use in the transformation
 * @return the transformed string (`[[nodiscard]]`)
 */
[[nodiscard]] std::string as_upper_case(std::string_view str);

/**
 * @brief Split the string @p str at the positions with delimiter @p delim and return the sub-strings.
 * @param[in] str the string to split
 * @param[in] delim the split delimiter
 * @return the split sub-strings (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<std::string_view> split(std::string_view str, char delim = ' ');

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_STRING_UTILITY_HPP_