/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different solver types.
 */

#include "plssvm/solver_types.hpp"

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::solver_type -> std::string conversions are correct
TEST(SolverType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::solver_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::solver_type::cg_explicit, "cg_explicit");
    EXPECT_CONVERSION_TO_STRING(plssvm::solver_type::cg_streaming, "cg_streaming");
    EXPECT_CONVERSION_TO_STRING(plssvm::solver_type::cg_implicit, "cg_implicit");
}
TEST(SolverType, to_string_unknown) {
    // check conversions to std::string from unknown solver_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::solver_type>(4), "unknown");
}

// check whether the std::string -> plssvm::solver_type conversions are correct
TEST(SolverType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::solver_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOmatic", plssvm::solver_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("cg_explicit", plssvm::solver_type::cg_explicit);
    EXPECT_CONVERSION_FROM_STRING("CG_Explicit", plssvm::solver_type::cg_explicit);
    EXPECT_CONVERSION_FROM_STRING("cg_streaming", plssvm::solver_type::cg_streaming);
    EXPECT_CONVERSION_FROM_STRING("CG_Streaming", plssvm::solver_type::cg_streaming);
    EXPECT_CONVERSION_FROM_STRING("cg_implicit", plssvm::solver_type::cg_implicit);
    EXPECT_CONVERSION_FROM_STRING("CG_Implicit", plssvm::solver_type::cg_implicit);
}
TEST(SolverType, from_string_unknown) {
    // foo isn't a valid solver_type
    std::istringstream input{ "foo" };
    plssvm::solver_type solver{};
    input >> solver;
    EXPECT_TRUE(input.fail());
}