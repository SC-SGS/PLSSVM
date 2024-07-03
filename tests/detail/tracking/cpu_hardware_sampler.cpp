/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the CPU hardware sampler class.
 */

#include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"  // plssvm::detail::tracking::cpu_hardware_sampler
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::number_of_substring_occurrences

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_FALSE

#include <chrono>  // std::chrono_literals namespace, std::chrono::steady_clock
#include <string>  // std::string
#include <tuple>   // std::ignore

using namespace std::chrono_literals;

TEST(CPUHardwareSampler, construct) {
    // construct a new CPU hardware sampler
    const plssvm::detail::tracking::cpu_hardware_sampler sampler{ 75ms };

    // no sampling should have started yet
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // check the sampling interval
    EXPECT_EQ(sampler.sampling_interval(), 75ms);
}

TEST(CPUHardwareSampler, construct_default_interval) {
    // construct a new CPU hardware sampler
    const plssvm::detail::tracking::cpu_hardware_sampler sampler{};

    // no sampling should have started yet
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // check the sampling interval
    EXPECT_EQ(sampler.sampling_interval(), PLSSVM_HARDWARE_SAMPLING_INTERVAL);
}

TEST(CPUHardwareSampler, yaml_string) {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // construct a new CPU hardware sampler
    plssvm::detail::tracking::cpu_hardware_sampler sampler{};

    // start sampling and then stop it again
    sampler.start_sampling();
    ASSERT_TRUE(sampler.is_sampling());
    sampler.stop_sampling();

    // check the YAML output string
    const std::string yaml_string = sampler.generate_yaml_string(start_time);

    // the yaml string may not be empty!
    ASSERT_FALSE(yaml_string.empty());
    // some strings MUST be in the YAML string
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("sampling_interval"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("time_points"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("general"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("clock"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("power"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("memory"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("temperature"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("integrated_gpu"));
    EXPECT_THAT(yaml_string, ::testing::HasSubstr("idle_states"));
    // unit and values must occur exactly the same number of times
    EXPECT_EQ(util::number_of_substring_occurrences(yaml_string, "unit"), util::number_of_substring_occurrences(yaml_string, "values"));
}

TEST(CPUHardwareSampler, yaml_string_still_sampling) {
    // construct a new CPU hardware sampler
    plssvm::detail::tracking::cpu_hardware_sampler sampler{};

    // start sampling
    sampler.start_sampling();
    ASSERT_TRUE(sampler.is_sampling());

    // can't generate the YAML string while the hardware sampler is still sampling
    EXPECT_THROW_WHAT(std::ignore = sampler.generate_yaml_string(std::chrono::steady_clock::now()), plssvm::hardware_sampling_exception, "Can't create the final YAML entry if the hardware sampler is still running!");
}

TEST(CPUHardwareSampler, device_identification) {
    // construct a new CPU hardware sampler
    const plssvm::detail::tracking::cpu_hardware_sampler sampler{};

    // check the device identification string
    EXPECT_EQ(sampler.device_identification(), std::string{ "cpu_device" });
}

TEST(CPUHardwareSampler, sampling_target) {
    // construct a new CPU hardware sampler
    const plssvm::detail::tracking::cpu_hardware_sampler sampler{};

    // check the target platform
    EXPECT_EQ(sampler.sampling_target(), plssvm::target_platform::cpu);
}
