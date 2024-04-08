/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "fmt/core.h"  // fmt::format

#include <string>         // std::string
#include <unordered_map>  // std::unordered_map

namespace plssvm::stdpar::detail {

std::string get_stdpar_version() {
    // create version map according to https://en.cppreference.com/w/cpp/preprocessor/replace
    // clang-format off
    static const std::unordered_map<long long, std::string> version_map{
        { 199711L, "c++98" },
        { 201103L, "c++11" },
        { 201402L, "c++14" },
        { 201703L, "c++17" },
        { 202002L, "c++20" },
        { 202302L, "c++23" }
    };
    // clang-format on

    // return sanitized version or plain __cplusplus if the version isn't found
    if (::plssvm::detail::contains(version_map, __cplusplus)) {
        return version_map.at(__cplusplus);
    } else {
        return fmt::format("{}", __cplusplus);
    }
}

}  // namespace plssvm::stdpar::detail
