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

#include "../../utility.hpp"  // util::redirect_output

#include "gtest/gtest.h"  // :testing::Test

#include <cstring>      // std::strcpy
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace util {

/**
 * @brief Fixture class for testing the parameter_* classes' implementation.
 */
class ParameterBase : public ::testing::Test, private redirect_output<> {
  protected:
    /**
     * @brief Create artificial argc and argv from the given command line string.
     * @param[in] cmd_line_split the command line argument to create the argc and argv from
     */
    void CreateCMDArgs(const std::vector<std::string> &cmd_line_split) {
        // create argc and argv from a std::string
        argc = static_cast<int>(cmd_line_split.size());
        argv = new char *[argc];
        for (int i = 0; i < argc; ++i) {
            argv[i] = new char[cmd_line_split[i].size() + 1];
            std::strcpy(argv[i], cmd_line_split[i].c_str());
        }
    }
    /**
     * @brief Free memory used for argv and end capturing std::cout. Automatically called at the end of a test.
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

}  // namespace util

#endif  // PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
