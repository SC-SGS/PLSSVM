/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the Kokkos backend.
 */

#include "plssvm/backends/Kokkos/detail/utility.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // ::testing::StartsWith
#include "gtest/gtest.h"  // TEST, EXPECT_GE, EXPECT_NO_THROW

#include <regex>  // std::regex, std::regex::extended, std::regex_match
