/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the logging function that includes tracking.
 */

#include "plssvm/detail/logging/log_untracked.hpp"

#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity, plssvm::verbosity_level

#include "tests/utility.hpp"  // util::redirect_output

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST_F, EXPECT_EQ, EXPECT_TRUE, ::testing::Test

class LoggerUntracked : public ::testing::Test,
                        public util::redirect_output<> { };

TEST_F(LoggerUntracked, EnabledLogging) {
    // explicitly enable logging
    plssvm::verbosity = plssvm::verbosity_level::full;

    // log a message
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "Hello, World!");

    // check captured output
    EXPECT_EQ(this->get_capture(), "Hello, World!");
}

TEST_F(LoggerUntracked, EnabledLoggingWithArgs) {
    // explicitly enable logging
    plssvm::verbosity = plssvm::verbosity_level::full;

    // log a message
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // check captured output
    EXPECT_EQ(this->get_capture(), "int: 42, float: 1.5, str: abc");
}

TEST_F(LoggerUntracked, DisabledLogging) {
    // explicitly disable logging
    plssvm::verbosity = plssvm::verbosity_level::quiet;

    // log message
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "Hello, World!");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}

TEST_F(LoggerUntracked, DisabledLoggingWithArgs) {
    // explicitly disable logging
    plssvm::verbosity = plssvm::verbosity_level::quiet;

    // log message
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}

TEST_F(LoggerUntracked, MismatchingVerbosityLevel) {
    // set verbosity_level to libsvm
    plssvm::verbosity = plssvm::verbosity_level::libsvm;

    // log message with full
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "Hello, World!");
    plssvm::detail::log_untracked(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // there should not be any output
    EXPECT_TRUE(this->get_capture().empty());
}

class WarningLoggerUntracked : public ::testing::Test,
                               public util::redirect_output<&std::clog> { };

TEST_F(WarningLoggerUntracked, EnabledLoggingWarning) {
    // explicitly enable logging
    plssvm::verbosity = plssvm::verbosity_level::full;

    // log a message
    plssvm::detail::log_untracked(plssvm::verbosity_level::warning, "WARNING!");

    // check captured output
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING!"));
}
