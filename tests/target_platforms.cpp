/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different target_platforms.
 */

#include "plssvm/target_platforms.hpp"

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "utility.hpp"  // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_GE

#include <sstream>  // std::istringstream
#include <vector>   // std::vector

// check whether the plssvm::target_platform -> std::string conversions are correct
TEST(TargetPlatform, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::target_platform::automatic), "automatic");
    EXPECT_EQ(util::convert_to_string(plssvm::target_platform::cpu), "cpu");
    EXPECT_EQ(util::convert_to_string(plssvm::target_platform::gpu_nvidia), "gpu_nvidia");
    EXPECT_EQ(util::convert_to_string(plssvm::target_platform::gpu_amd), "gpu_amd");
    EXPECT_EQ(util::convert_to_string(plssvm::target_platform::gpu_intel), "gpu_intel");
}
TEST(TargetPlatform, to_string_unknown) {
    // check conversions to std::string from unknown target_platform
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::target_platform>(5)), "unknown");
}

// check whether the std::string -> plssvm::target_platform conversions are correct
TEST(TargetPlatform, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("automatic"), plssvm::target_platform::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("AUTOmatic"), plssvm::target_platform::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("cpu"), plssvm::target_platform::cpu);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("CPU"), plssvm::target_platform::cpu);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("gpu_nvidia"), plssvm::target_platform::gpu_nvidia);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("GPU_NVIDIA"), plssvm::target_platform::gpu_nvidia);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("gpu_amd"), plssvm::target_platform::gpu_amd);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("GPU_AMD"), plssvm::target_platform::gpu_amd);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("gpu_intel"), plssvm::target_platform::gpu_intel);
    EXPECT_EQ(util::convert_from_string<plssvm::target_platform>("GPU_INTEL"), plssvm::target_platform::gpu_intel);
}
TEST(TargetPlatform, from_string_unknown) {
    // foo isn't a valid target_platform
    std::istringstream input{ "foo" };
    plssvm::target_platform platform;
    input >> platform;
    EXPECT_TRUE(input.fail());
}

TEST(TargetPlatform, minimal_available_target_platform) {
    // get the available target platforms
    const std::vector<plssvm::target_platform> platform = plssvm::list_available_target_platforms();

    // at least two target platforms must be available (automatic + one user provided)!
    EXPECT_GE(platform.size(), 2);

    // the automatic backend must always be present
    EXPECT_TRUE(plssvm::detail::contains(platform, plssvm::target_platform::automatic));
}