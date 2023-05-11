/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom PLSSVM_ASSERT implementation.
 */

#include "plssvm/detail/assert.hpp"

#include "gmock/gmock-matchers.h"  // ::testing::ContainsRegex
#include "gtest/gtest.h"           // TEST, ASSERT_DEATH, EXPECT_DEATH

// only test if assertions are enabled
#if defined(PLSSVM_ASSERT_ENABLED)

TEST(PLSSVMAssert, assert_true) {
    // must not trigger an assertion
    PLSSVM_ASSERT(true, "TRUE");
}
TEST(PLSSVMAssert, assert_false) {
    ASSERT_DEATH(PLSSVM_ASSERT(false, "FALSE"), ::testing::ContainsRegex("Assertion '.*false.*' failed!"));
}

#endif

// check the internal check_assertion function
TEST(PLSSVMAssert, check_assertion_true) {
    // calling check assertion with true shouldn't do anything
    plssvm::detail::check_assertion(true, "", plssvm::source_location::current(), "");
}
TEST(PLSSVMAssert, check_assertion_false) {
    // calling check assertion with false should abort
    EXPECT_DEATH(plssvm::detail::check_assertion(false, "cond", plssvm::source_location::current(), "msg {}", 1), "cond");
}