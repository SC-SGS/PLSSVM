/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom PLSSVM_ASSERT implementation.
 */

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, ASSERT_DEATH

// only test if assertions are enabled
#if defined(PLSSVM_ASSERT_ENABLED)

TEST(PLSSVMAssert, assert_true) {
    // must not trigger an assertion
    PLSSVM_ASSERT(true, "TRUE");
}

TEST(PLSSVMAssert, assert_false) {
    // can't use a regex matcher due to the used emphasis and color specification in the assertion message
    ASSERT_DEATH(PLSSVM_ASSERT(false, "FALSE"), ::testing::HasSubstr("FALSE"));
}

#endif