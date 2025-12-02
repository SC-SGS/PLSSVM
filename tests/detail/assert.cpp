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

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#if defined(PLSSVM_ENABLE_ASSERTS)
    #include "fmt/format.h"  // fmt::format

    #include <string>  // std::string
#endif

#include "gtest/gtest.h"  // TEST, ASSERT_DEATH, EXPECT_DEATH, ::testing::ContainsRegex

// only test if assertions are enabled
#if defined(PLSSVM_ENABLE_ASSERTS)

TEST(PLSSVMAssert, AssertTrue) {
    // must not trigger an assertion
    PLSSVM_ASSERT(true, "TRUE");
}

TEST(PLSSVMAssert, AssertFalse) {
    ASSERT_DEATH(PLSSVM_ASSERT(false, "FALSE"), ::testing::ContainsRegex("Assertion '.*false.*' failed!"));
}

TEST(PLSSVMAssertDeathTest, CheckAssertionFalse) {
    const auto loc = plssvm::source_location::current();

    // test regex
    const std::string regex = fmt::format("Assertion '.*1 == 2.*' failed!\n"
                                          "{}"
                                          "  in file            .*\n"
                                          "  in function        .*\n"
                                          "  @ line             .*\n\n"
                                          ".*msg 1.*\n",
                                          loc.world_rank().has_value() ? "  on MPI world rank  .*\n" : "");

    // calling check assertion with false should abort
    EXPECT_DEATH(plssvm::detail::check_assertion(1 == 2, "1 == 2", loc, "msg {}", 1), ::testing::ContainsRegex(regex));
}

#endif

// check the internal check_assertion function
TEST(PLSSVMAssert, CheckAssertionTrue) {
    // calling check assertion with true shouldn't do anything
    plssvm::detail::check_assertion(true, "", plssvm::source_location::current(), "");
}
