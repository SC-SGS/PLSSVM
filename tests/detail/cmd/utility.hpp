/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions and classes (fixtures) for testing the parameter_* classes' functionality.
 */

#ifndef PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
#define PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
#pragma once

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::split_as
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::replace_all

#include "../../utility.hpp"  // util::redirect_output

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // :testing::TestParamInfo

#include <cstring>      // std::strcpy
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::get
#include <vector>       // std::vector

namespace util {

/**
 * Fixture class for testing the parameter_* classes' implementation.
 */
class ParameterBase : public ::testing::Test, private redirect_output {
  protected:
    /**
     * @brief Create artificial argc and argv from the given string.
     * @param[in] cmd_line the command line argument to create the argc and argv from.
     */
    virtual void CreateCMDArgs(const std::string_view cmd_line) {
        // create argc and argv from a std::string
        const std::vector<std::string> cmd_line_split = plssvm::detail::split_as<std::string>(cmd_line);
        argc = static_cast<int>(cmd_line_split.size());
        argv = new char *[argc];
        for (int i = 0; i < argc; ++i) {
            argv[i] = new char[cmd_line_split[i].size() + 1];
            std::strcpy(argv[i], cmd_line_split[i].c_str());
        }
    }
    /**
     * Free memory used for argv and end capturing std::cout. Automatically called at the end of a test.
     */
    void TearDown() override {
        // free memory at the end
        for (int i = 0; i < argc; ++i) {
            delete[] argv[i];
        }
        delete[] argv;
    }

    /// The number of the artificial command line arguments.
    int argc{ 0 };
    /// The artificial command line arguments.
    char **argv{ nullptr };
};

// pretty printer
/**
 * @brief Pretty print a flag and value combination.
 * @details Replaces all "-" in a flag with "", all "-" in a value with "m" (for minus), and all "." in a value with "p" (for point).
 * @tparam T the parameter type used in the test fixture
 * @param[in] param_info the parameter info used for pretty printing the test case name
 * @return the test case name
 */
template <typename T>
const auto pretty_print_flag_and_value = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = std::get<0>(param_info.param);
    plssvm::detail::replace_all(flag, "-", "");
    // sanitize values for Google Test names
    std::string value = fmt::format("{}", std::get<1>(param_info.param));
    plssvm::detail::replace_all(value, "-", "m");
    plssvm::detail::replace_all(value, ".", "p");
    return fmt::format("{}_{}", flag, value);
};
/**
 * @brief Pretty print a flag.
 * @details Replaces all "-" in a flag with "".
 * @tparam T the parameter type used in the test fixture
 * @param[in] param_info the parameter info used for pretty printing the test case name
 * @return the test case name
 */
template <typename T>
const auto pretty_print_flag = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = param_info.param;
    plssvm::detail::replace_all(flag, "-", "");
    return fmt::format("{}", flag.empty() ? "EMPTY_FLAG" : flag);
};

}  // namespace util

#endif  // PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
