/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the logging function.
 */

#include "plssvm/detail/logger.hpp"

#include "plssvm/constants.hpp"  // plssvm::verbose

#include "../utility.hpp"  // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_EQ, EXPECT_TRUE, ::testing::Test

class Logger : public ::testing::Test, public util::redirect_output<> {};

TEST_F(Logger, enabled_logging) {
    // explicitly enable logging
    plssvm::verbose = true;

    // log a message
    plssvm::detail::log("Hello, World!");

    // check captured output
    EXPECT_EQ(this->get_capture(), "Hello, World!");
}
TEST_F(Logger, enabled_logging_with_args) {
    // explicitly enable logging
    plssvm::verbose = true;

    // log a message
    plssvm::detail::log("int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // check captured output
    EXPECT_EQ(this->get_capture(), "int: 42, float: 1.5, str: abc");
}

TEST_F(Logger, disabled_logging) {
    // explicitly disable logging
    plssvm::verbose = false;

    // log message
    plssvm::detail::log("Hello, World!");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}
TEST_F(Logger, disabled_logging_with_args) {
    // explicitly disable logging
    plssvm::verbose = false;

    // log message
    plssvm::detail::log("int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}