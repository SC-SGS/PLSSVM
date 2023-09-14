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

#include "plssvm/detail/utility.hpp"  // plssvm::detail::to_underlying

#include "../custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING
#include "../utility.hpp"             // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_EQ, EXPECT_TRUE, ::testing::Test

#include <sstream>  // std::istringstream

// check whether the plssvm::verbosity_level values are power of twos
TEST(VerbosityLevel, values) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::quiet), 0b000);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::libsvm), 0b001);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::timing), 0b010);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full), 0b100);
}
// check whether the plssvm::verbosity_level -> std::string conversions are correct
TEST(VerbosityLevel, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::quiet, "quiet");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::libsvm, "libsvm");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::timing, "timing");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full, "full");
}
TEST(VerbosityLevel, to_string_concatenation) {
    // check conversion to std::string for multiple values
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm,
                                "libsvm | timing | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                                "timing | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm,
                                "libsvm | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm,
                                "libsvm | timing");
}
TEST(VerbosityLevel, to_string_unknown) {
    // check conversions to std::string from unknown backend_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::verbosity_level>(0b1000), "unknown");
}

// check whether the std::string -> plssvm::verbosity_level conversions are correct
TEST(VerbosityLevel, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("quiet", plssvm::verbosity_level::quiet);
    EXPECT_CONVERSION_FROM_STRING("QUIET", plssvm::verbosity_level::quiet);
    EXPECT_CONVERSION_FROM_STRING("libsvm", plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("LIBSVM", plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("timing", plssvm::verbosity_level::timing);
    EXPECT_CONVERSION_FROM_STRING("TIMING", plssvm::verbosity_level::timing);
    EXPECT_CONVERSION_FROM_STRING("full", plssvm::verbosity_level::full);
    EXPECT_CONVERSION_FROM_STRING("FULL", plssvm::verbosity_level::full);
}
TEST(VerbosityLevel, from_string_concatenation) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("quiet|libsvm|timing|full", plssvm::verbosity_level::quiet);
    EXPECT_CONVERSION_FROM_STRING("libsvm|timing|full", plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("timing|full", plssvm::verbosity_level::full | plssvm::verbosity_level::timing);
    EXPECT_CONVERSION_FROM_STRING("libsvm|full", plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("libsvm|timing", plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm);
}
TEST(VerbosityLevel, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream input{ "foo" };
    plssvm::verbosity_level verb{};
    input >> verb;
    EXPECT_TRUE(input.fail());
}

TEST(VerbosityLevel, bitwise_or) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm), 0b111);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::timing), 0b110);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm), 0b101);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm), 0b011);
}
TEST(VerbosityLevel, compound_bitwise_or) {
    plssvm::verbosity_level verb = plssvm::verbosity_level::full;
    verb |= plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm;
    EXPECT_EQ(plssvm::detail::to_underlying(verb), 0b111);
}
TEST(VerbosityLevel, bitwise_and) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full & plssvm::verbosity_level::full), 0b100);
    const plssvm::verbosity_level verb = plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm;
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::quiet), 0b000);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::libsvm), 0b001);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::timing), 0b000);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::full), 0b100);
}
TEST(VerbosityLevel, compound_bitwise_and) {
    plssvm::verbosity_level verb = plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm;
    verb &= plssvm::verbosity_level::full;
    EXPECT_EQ(plssvm::detail::to_underlying(verb), 0b100);
}

class Logger : public ::testing::Test, public util::redirect_output<> {};

TEST_F(Logger, enabled_logging) {
    // explicitly enable logging
    plssvm::verbosity = plssvm::verbosity_level::full;

    // log a message
    plssvm::detail::log(plssvm::verbosity_level::full, "Hello, World!");

    // check captured output
    EXPECT_EQ(this->get_capture(), "Hello, World!");
}
TEST_F(Logger, enabled_logging_with_args) {
    // explicitly enable logging
    plssvm::verbosity = plssvm::verbosity_level::full;

    // log a message
    plssvm::detail::log(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // check captured output
    EXPECT_EQ(this->get_capture(), "int: 42, float: 1.5, str: abc");
}

TEST_F(Logger, disabled_logging) {
    // explicitly disable logging
    plssvm::verbosity = plssvm::verbosity_level::quiet;

    // log message
    plssvm::detail::log(plssvm::verbosity_level::full, "Hello, World!");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}
TEST_F(Logger, disabled_logging_with_args) {
    // explicitly disable logging
    plssvm::verbosity = plssvm::verbosity_level::quiet;

    // log message
    plssvm::detail::log(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // since logging has been disabled, nothing should have been captured
    EXPECT_TRUE(this->get_capture().empty());
}

TEST_F(Logger, mismatching_verbosity_level) {
    // set verbosity_level to libsvm
    plssvm::verbosity = plssvm::verbosity_level::libsvm;

    // log message with full
    plssvm::detail::log(plssvm::verbosity_level::full, "Hello, World!");
    plssvm::detail::log(plssvm::verbosity_level::full, "int: {}, float: {}, str: {}", 42, 1.5, "abc");

    // there should not be any output
    EXPECT_TRUE(this->get_capture().empty());
}