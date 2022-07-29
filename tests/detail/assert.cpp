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

#include "gtest/gtest.h"  // TEST, ASSERT_DEATH

#if defined(PLSSVM_ASSERT_ENABLED)

// check whether the PLSSVM_ASSERT works correctly
TEST(BaseDeathTest, plssvm_assert) {
    PLSSVM_ASSERT(true, "TRUE");

    // can't use a matcher due to the used emphasis and color specification in assertion message
    ASSERT_DEATH(PLSSVM_ASSERT(false, "FALSE"), "");
}

#endif