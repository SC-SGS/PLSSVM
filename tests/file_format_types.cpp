/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different file format types.
 */

#include "plssvm/file_format_types.hpp"

#include "utility.hpp"  // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::file_format_type -> std::string conversions are correct
TEST(FileFormatType, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::file_format_type::libsvm), "libsvm");
    EXPECT_EQ(util::convert_to_string(plssvm::file_format_type::arff), "arff");
}
TEST(FileFormatType, to_string_unknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::file_format_type>(2)), "unknown");
}

// check whether the std::string -> plssvm::file_format_type conversions are correct
TEST(FileFormatType, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::file_format_type>("LIBSVM"), plssvm::file_format_type::libsvm);
    EXPECT_EQ(util::convert_from_string<plssvm::file_format_type>("libsvm"), plssvm::file_format_type::libsvm);
    EXPECT_EQ(util::convert_from_string<plssvm::file_format_type>("ARFF"), plssvm::file_format_type::arff);
    EXPECT_EQ(util::convert_from_string<plssvm::file_format_type>("arff"), plssvm::file_format_type::arff);
}
TEST(FileFormatType, from_string_unknown) {
    // foo isn't a valid file_format_type
    std::istringstream ss{ "foo" };
    plssvm::file_format_type ff;
    ss >> ff;
    EXPECT_TRUE(ss.fail());
}
