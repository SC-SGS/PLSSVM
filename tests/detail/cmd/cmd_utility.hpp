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

#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity_level, plssvm::verbosity

#include "tests/utility.hpp"  // util::redirect_output

#include "gtest/gtest.h"  // :testing::Test

#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace util {

/**
 * @brief Fixture class for testing the parameter_* classes' implementation.
 */
class ParameterBase : public ::testing::Test,
                      private redirect_output<> {
  protected:
    void SetUp() override {
        // save the current verbosity state
        verbosity_save_ = plssvm::verbosity;
    }

    /**
     * @brief Create artificial argc and argv from the given command line string.
     * @param[in] cmd_line_split the command line argument to create the argc and argv from
     */
    void CreateCMDArgs(std::vector<std::string> cmd_line_split) {
        // create argc and argv from a std::string
        cmd_options_ = std::move(cmd_line_split);
        cmd_argv_.reserve(cmd_options_.size());
        for (std::vector<std::string>::size_type i = 0; i < cmd_options_.size(); ++i) {
            cmd_argv_.push_back(cmd_options_[i].data());
        }
    }

    /**
     * @brief Reset the verbosity level; automatically called at the end of a test.
     */
    void TearDown() override {
        // restore verbosity state
        plssvm::verbosity = verbosity_save_;
    }

    /**
     * @brief Return the number of command line arguments encapsulated in this class.
     * @return the number of cmd arguments (`[[nodiscard]]`)
     */
    [[nodiscard]] int get_argc() const noexcept { return static_cast<int>(cmd_argv_.size()); }

    /**
     * @brief The command line arguments encapsulated in this class.
     * @return the cmd arguments (`[[nodiscard]])
     */
    [[nodiscard]] char **get_argv() const noexcept { return cmd_argv_.data(); }

  private:
    /// The provided command line options.
    mutable std::vector<std::string> cmd_options_{};
    /// The command line options cast to a char *.
    mutable std::vector<char *> cmd_argv_{};
    /// The verbosity level at the time of the test start.
    plssvm::verbosity_level verbosity_save_{};
};

}  // namespace util

#endif  // PLSSVM_TESTS_DETAIL_CMD_UTILITY_HPP_
