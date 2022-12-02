/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for functions related to the different SYCL implementation types.
*/

#include "plssvm/backends/SYCL/implementation_type.hpp"

#include "utility.hpp"  // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl_generic::implementation_type -> std::string conversions are correct
TEST(ImplementationType, to_string) {
   // check conversions to std::string
   EXPECT_EQ(util::convert_to_string(plssvm::sycl_generic::implementation_type::automatic), "automatic");
   EXPECT_EQ(util::convert_to_string(plssvm::sycl_generic::implementation_type::hipsycl), "hipsycl");
   EXPECT_EQ(util::convert_to_string(plssvm::sycl_generic::implementation_type::dpcpp), "dpcpp");
}
TEST(ImplementationType, to_string_unknown) {
   // check conversions to std::string from unknown file_format_type
   EXPECT_EQ(util::convert_to_string(static_cast<plssvm::sycl_generic::implementation_type>(3)), "unknown");
}

// check whether the std::string -> plssvm::sycl_generic::implementation_type conversions are correct
TEST(ImplementationType, from_string) {
   // check conversion from std::string
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("automatic"), plssvm::sycl_generic::implementation_type::automatic);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("AUTOMATIC"), plssvm::sycl_generic::implementation_type::automatic);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("hipsycl"), plssvm::sycl_generic::implementation_type::hipsycl);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("hipSYCL"), plssvm::sycl_generic::implementation_type::hipsycl);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("dpcpp"), plssvm::sycl_generic::implementation_type::dpcpp);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("DPCPP"), plssvm::sycl_generic::implementation_type::dpcpp);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("dpc++"), plssvm::sycl_generic::implementation_type::dpcpp);
   EXPECT_EQ(util::convert_from_string<plssvm::sycl_generic::implementation_type>("DPC++"), plssvm::sycl_generic::implementation_type::dpcpp);
}
TEST(ImplementationType, from_string_unknown) {
   // foo isn't a valid file_format_type
   std::istringstream input{ "foo" };
   plssvm::sycl_generic::implementation_type impl;
   input >> impl;
   EXPECT_TRUE(input.fail());
}
