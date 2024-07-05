/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the NVIDIA GPU hardware sampler class.
 */

#include "plssvm/detail/tracking/gpu_nvidia/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_nvidia_hardware_sampler
#include "plssvm/exceptions/exceptions.hpp"                        // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                             // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::number_of_substring_occurrences

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST_F, EXPECT_EQ, EXPECT_FALSE

#include <chrono>    // std::chrono_literals namespace, std::chrono::steady_clock
#include <iostream>  // std::cout
#include <string>    // std::string
#include <tuple>     // std::ignore

class GPUNVIDIAHardwareSampler : public ::testing::Test,
                                 protected util::redirect_output<> { };

using namespace std::chrono_literals;

TEST_F(GPUNVIDIAHardwareSampler, construct) {
    // construct a new NVIDIA GPU hardware sampler
    const plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0, 75ms };

    // no sampling should have started yet
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // check the sampling interval
    EXPECT_EQ(sampler.sampling_interval(), 75ms);
}

TEST_F(GPUNVIDIAHardwareSampler, construct_default_interval) {
    // construct a new NVIDIA GPU hardware sampler
    const plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // no sampling should have started yet
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // check the sampling interval
    EXPECT_EQ(sampler.sampling_interval(), PLSSVM_HARDWARE_SAMPLING_INTERVAL);
}

TEST_F(GPUNVIDIAHardwareSampler, yaml_string) {
    const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // construct a new NVIDIA GPU hardware sampler
    plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

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
    // unit and values must occur exactly the same number of times
    EXPECT_EQ(util::number_of_substring_occurrences(yaml_string, "unit"), util::number_of_substring_occurrences(yaml_string, "values"));
}

TEST_F(GPUNVIDIAHardwareSampler, yaml_string_still_sampling) {
    // construct a new NVIDIA GPU hardware sampler
    plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // start sampling
    sampler.start_sampling();
    ASSERT_TRUE(sampler.is_sampling());

    // can't generate the YAML string while the hardware sampler is still sampling
    EXPECT_THROW_WHAT(std::ignore = sampler.generate_yaml_string(std::chrono::steady_clock::now()), plssvm::hardware_sampling_exception, "Can't create the final YAML entry if the hardware sampler is still running!");
}

TEST_F(GPUNVIDIAHardwareSampler, device_identification) {
    // construct a new NVIDIA GPU hardware sampler
    const plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // check the device identification string
    EXPECT_THAT(sampler.device_identification(), ::testing::StartsWith("gpu_nvidia_device_"));
}

TEST_F(GPUNVIDIAHardwareSampler, sampling_target) {
    // construct a new NVIDIA GPU hardware sampler
    const plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // check the target platform
    EXPECT_EQ(sampler.sampling_target(), plssvm::target_platform::gpu_nvidia);
}

TEST_F(GPUNVIDIAHardwareSampler, output_operator) {
    // construct a new NVIDIA GPU hardware sampler
    plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // start sampling and then stop it again
    sampler.start_sampling();
    ASSERT_TRUE(sampler.is_sampling());
    sampler.stop_sampling();

    // get the output string
    std::cout << sampler;
    const std::string str = this->get_capture();

    ASSERT_FALSE(std::cout.fail());

    // the output string may not be empty!
    ASSERT_FALSE(str.empty());
    // some strings MUST be in the output string
    EXPECT_THAT(str, ::testing::HasSubstr("sampling interval"));
    EXPECT_THAT(str, ::testing::HasSubstr("time points"));
    EXPECT_THAT(str, ::testing::HasSubstr("general samples"));
    EXPECT_THAT(str, ::testing::HasSubstr("clock samples"));
    EXPECT_THAT(str, ::testing::HasSubstr("power samples"));
    EXPECT_THAT(str, ::testing::HasSubstr("memory samples"));
    EXPECT_THAT(str, ::testing::HasSubstr("temperature samples"));
}

TEST_F(GPUNVIDIAHardwareSampler, output_operator_still_sampling) {
    // construct a new NVIDIA GPU hardware sampler
    plssvm::detail::tracking::gpu_nvidia_hardware_sampler sampler{ 0 };

    // start sampling and then stop it again
    sampler.start_sampling();
    ASSERT_TRUE(sampler.is_sampling());

    // get the output string -> will fail if the sampler is currently sampling
    std::cout << sampler;
    ASSERT_TRUE(std::cout.fail());
}