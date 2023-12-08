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
#include "plssvm/verbosity_levels.hpp"          // plssvm::verbosity_level, plssvm::verbosity

#include "utility.hpp"  // util::redirect_output

#include "gtest/gtest.h"  // :testing::Test

#include <cstring>      // std::memcpy
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace util {

/**
 * @brief Fixture class for testing the parameter_* classes' implementation.
 */
class ParameterBase : public ::testing::Test, private redirect_output<> {
  protected:
    void SetUp() override {
        // save the current verbosity state
        verbosity_save_ = plssvm::verbosity;
    }
    /**
     * @brief Create artificial argc and argv from the given command line string.
     * @param[in] cmd_line_split the command line argument to create the argc and argv from
     */
    void CreateCMDArgs(const std::vector<std::string> &cmd_line_split) {
        // create argc and argv from a std::string
        argc_ = static_cast<int>(cmd_line_split.size());
        argv_ = new char *[argc_];
        for (int i = 0; i < argc_; ++i) {
            const std::size_t arg_size = cmd_line_split[i].size() + 1;
            argv_[i] = new char[arg_size];
            std::memcpy(argv_[i], cmd_line_split[i].c_str(), arg_size * sizeof(char));
        }
    }
    /**
     * @brief Free memory used for argv; automatically called at the end of a test.
     */
    void TearDown() override {
        // free memory at the end
        for (int i = 0; i < argc_; ++i) {
            delete[] argv_[i];
        }
        delete[] argv_;
        // restore verbosity state
        plssvm::verbosity = verbosity_save_;
    }

    /**
     * @brief Return the number of command line arguments encapsulated in this class.
     * @return the number of cmd arguments (`[[nodiscard]]`)
     */
    [[nodiscard]] int get_argc() const noexcept { return argc_; }
    /**
     * @brief The command line arguments encapsulated in this class.
     * @return the cmd arguments (`[[nodiscard]])
     */
    [[nodiscard]] char **get_argv() const noexcept { return argv_; }

  private:
    /// The number of the artificial command line arguments.
    int argc_{ 0 };
    /// The artificial command line arguments.
    char **argv_{ nullptr };
    /// The verbosity level at the time of the test start.
    plssvm::verbosity_level verbosity_save_{};
};

}  // namespace util

#endif  // PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
