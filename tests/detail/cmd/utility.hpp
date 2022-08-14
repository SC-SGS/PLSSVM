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

#include "gtest/gtest.h"  // ASSERT_FALSE, ::testing::Test

#include <cstring>      // std::strcpy
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream, std::ostringstream, std::istringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace util {

/*
 * Fixture class for testing the parameter_* classes' implementation.
 */
class ParameterBase : public ::testing::Test {
  protected:
    /*
     * Create artificial argc and argv from the given string.
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
    void SetUp() override {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }
    /*
     * Free memory used for argv. Automatically called at the end of a test.
     */
    void TearDown() override {
        // free memory at the end
        for (int i = 0; i < argc; ++i) {
            delete[] argv[i];
        }
        delete[] argv;

        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

    // The number of the artificial command line arguments.
    int argc{ 0 };
    // The artificial command line arguments.
    char **argv{ nullptr };

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};

}  // namespace util

#endif  // PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
