/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the verbosity level enumeration.
 */

#include "plssvm/verbosity_levels.hpp"

#include "plssvm/detail/utility.hpp"  // plssvm::detail::to_underlying

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING
#include "utility.hpp"             // util::redirect_output

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::verbosity_level values are power of twos
TEST(VerbosityLevel, values) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::quiet), 0b0000);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::libsvm), 0b0001);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::timing), 0b0010);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::warning), 0b0100);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full), 0b1000);
}
// check whether the plssvm::verbosity_level -> std::string conversions are correct
TEST(VerbosityLevel, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::quiet, "quiet");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::libsvm, "libsvm");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::timing, "timing");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::warning, "warning");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full, "full");
}
TEST(VerbosityLevel, to_string_concatenation) {
    // check conversion to std::string for multiple values
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm,
                                "libsvm | timing | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::warning | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm,
                                "libsvm | timing | warning | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
                                "timing | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm,
                                "libsvm | full");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm,
                                "libsvm | timing");
    EXPECT_CONVERSION_TO_STRING(plssvm::verbosity_level::warning | plssvm::verbosity_level::libsvm,
                                "libsvm | warning");
}
TEST(VerbosityLevel, to_string_unknown) {
    // check conversions to std::string from unknown backend_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::verbosity_level>(0b10000), "unknown");
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
    EXPECT_CONVERSION_FROM_STRING("warning", plssvm::verbosity_level::warning);
    EXPECT_CONVERSION_FROM_STRING("WARNING", plssvm::verbosity_level::warning);
    EXPECT_CONVERSION_FROM_STRING("full", plssvm::verbosity_level::full);
    EXPECT_CONVERSION_FROM_STRING("FULL", plssvm::verbosity_level::full);
}
TEST(VerbosityLevel, from_string_concatenation) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("quiet|libsvm|timing|full", plssvm::verbosity_level::quiet);
    EXPECT_CONVERSION_FROM_STRING("libsvm|timing|warning|full", plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::warning | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("libsvm|timing|full", plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("timing|full", plssvm::verbosity_level::full | plssvm::verbosity_level::timing);
    EXPECT_CONVERSION_FROM_STRING("libsvm|full", plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("libsvm|timing", plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm);
    EXPECT_CONVERSION_FROM_STRING("libsvm|warning", plssvm::verbosity_level::warning | plssvm::verbosity_level::libsvm);
}
TEST(VerbosityLevel, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream input{ "foo" };
    plssvm::verbosity_level verb{};
    input >> verb;
    EXPECT_TRUE(input.fail());
}

TEST(VerbosityLevel, bitwise_or) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::warning | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm), 0b1111);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm), 0b1011);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::timing), 0b1010);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm), 0b1001);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm), 0b0011);
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::warning | plssvm::verbosity_level::libsvm), 0b0101);
}
TEST(VerbosityLevel, compound_bitwise_or) {
    plssvm::verbosity_level verb = plssvm::verbosity_level::full;
    verb |= plssvm::verbosity_level::timing | plssvm::verbosity_level::libsvm;
    EXPECT_EQ(plssvm::detail::to_underlying(verb), 0b1011);
}
TEST(VerbosityLevel, bitwise_and) {
    EXPECT_EQ(plssvm::detail::to_underlying(plssvm::verbosity_level::full & plssvm::verbosity_level::full), 0b1000);
    const plssvm::verbosity_level verb = plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm;
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::quiet), 0b0000);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::libsvm), 0b0001);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::timing), 0b0000);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::warning), 0b0000);
    EXPECT_EQ(plssvm::detail::to_underlying(verb & plssvm::verbosity_level::full), 0b1000);
}
TEST(VerbosityLevel, compound_bitwise_and) {
    plssvm::verbosity_level verb = plssvm::verbosity_level::full | plssvm::verbosity_level::libsvm;
    verb &= plssvm::verbosity_level::full;
    EXPECT_EQ(plssvm::detail::to_underlying(verb), 0b1000);
}