/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom source_location implementation based on C++20's [`std::source_location`](https://en.cppreference.com/w/cpp/utility/source_location).
 */

#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include "plssvm/mpi/environment.hpp"  // plssvm::mpi::is_active

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ

#include <cstdint>  // std::uint_least32_t

// dummy function to be able to specify the function name
[[nodiscard]] plssvm::source_location dummy() {
    return plssvm::source_location::current();
}

TEST(SourceLocation, DefaultConstruct) {
    const plssvm::source_location loc{};

    EXPECT_EQ(loc.file_name(), std::string{ "unknown" });
    EXPECT_EQ(loc.function_name(), std::string{ "unknown" });
    EXPECT_EQ(loc.line(), std::uint_least32_t{ 0 });
    EXPECT_EQ(loc.column(), std::uint_least32_t{ 0 });
    EXPECT_FALSE(loc.world_rank().has_value());
}

TEST(SourceLocation, CurrentLocation) {
    const plssvm::source_location loc = dummy();

    EXPECT_EQ(loc.file_name(), __builtin_FILE());
    EXPECT_THAT(loc.function_name(), ::testing::HasSubstr("dummy"));
    EXPECT_EQ(loc.line(), std::uint_least32_t{ 22 });   // attention: hardcoded line!
    EXPECT_EQ(loc.column(), std::uint_least32_t{ 0 });  // attention: always 0!

    if (plssvm::mpi::is_active()) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        ASSERT_TRUE(loc.world_rank().has_value());
        // since MPI is disabled, the world rank must always be zero
        EXPECT_EQ(loc.world_rank().value(), 0);
#else
        EXPECT_FALSE(loc.world_rank().has_value());
#endif
    } else {
        EXPECT_FALSE(loc.world_rank().has_value());
    }
}
