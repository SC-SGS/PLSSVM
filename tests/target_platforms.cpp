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

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_GE

#include <sstream>  // std::istringstream
#include <vector>   // std::vector

// check whether the plssvm::target_platform -> std::string conversions are correct
TEST(TargetPlatform, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::target_platform::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::target_platform::cpu, "cpu");
    EXPECT_CONVERSION_TO_STRING(plssvm::target_platform::gpu_nvidia, "gpu_nvidia");
    EXPECT_CONVERSION_TO_STRING(plssvm::target_platform::gpu_amd, "gpu_amd");
    EXPECT_CONVERSION_TO_STRING(plssvm::target_platform::gpu_intel, "gpu_intel");
}
TEST(TargetPlatform, to_string_unknown) {
    // check conversions to std::string from unknown target_platform
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::target_platform>(5), "unknown");
}

// check whether the std::string -> plssvm::target_platform conversions are correct
TEST(TargetPlatform, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::target_platform::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOmatic", plssvm::target_platform::automatic);
    EXPECT_CONVERSION_FROM_STRING("cpu", plssvm::target_platform::cpu);
    EXPECT_CONVERSION_FROM_STRING("CPU", plssvm::target_platform::cpu);
    EXPECT_CONVERSION_FROM_STRING("gpu_nvidia", plssvm::target_platform::gpu_nvidia);
    EXPECT_CONVERSION_FROM_STRING("GPU_NVIDIA", plssvm::target_platform::gpu_nvidia);
    EXPECT_CONVERSION_FROM_STRING("gpu_amd", plssvm::target_platform::gpu_amd);
    EXPECT_CONVERSION_FROM_STRING("GPU_AMD", plssvm::target_platform::gpu_amd);
    EXPECT_CONVERSION_FROM_STRING("gpu_intel", plssvm::target_platform::gpu_intel);
    EXPECT_CONVERSION_FROM_STRING("GPU_INTEL", plssvm::target_platform::gpu_intel);
}
TEST(TargetPlatform, from_string_unknown) {
    // foo isn't a valid target_platform
    std::istringstream input{ "foo" };
    plssvm::target_platform platform{};
    input >> platform;
    EXPECT_TRUE(input.fail());
}

TEST(TargetPlatform, minimal_available_target_platform) {
    // get the available target platforms
    const std::vector<plssvm::target_platform> platform = plssvm::list_available_target_platforms();

    // at least two target platforms must be available (automatic + one user provided)!
    EXPECT_GE(platform.size(), 2);

    // the automatic backend must always be present
    EXPECT_THAT(platform, ::testing::Contains(plssvm::target_platform::automatic));
}

TEST(TargetPlatform, determine_default_target_platform) {
    // the determined default platform must not be target_platform::automatic
    const plssvm::target_platform target = plssvm::determine_default_target_platform();
    EXPECT_NE(target, plssvm::target_platform::automatic);
}
TEST(TargetPlatform, determine_target_platform) {
    // if only one platform is available, the default platform must be this platform
    EXPECT_EQ(plssvm::determine_default_target_platform({ plssvm::target_platform::gpu_nvidia }), plssvm::target_platform::gpu_nvidia);
    EXPECT_EQ(plssvm::determine_default_target_platform({ plssvm::target_platform::gpu_amd }), plssvm::target_platform::gpu_amd);
    EXPECT_EQ(plssvm::determine_default_target_platform({ plssvm::target_platform::gpu_intel }), plssvm::target_platform::gpu_intel);
    EXPECT_EQ(plssvm::determine_default_target_platform({ plssvm::target_platform::cpu }), plssvm::target_platform::cpu);
}