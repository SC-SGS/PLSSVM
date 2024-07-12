/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the hardware sampler factory function.
 */

#include "plssvm/detail/tracking/hardware_sampler_factory.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ

#include <memory>  // std::unique_ptr
#include <tuple>   // std::ignore

TEST(HardwareSamplerFactory, make_hardware_sampler_automatic) {
    switch (plssvm::determine_default_target_platform()) {
        case plssvm::target_platform::automatic:
            GTEST_FAIL() << "plssvm::target_platform::automatic not allowed as plssvm::determine_default_target_platform() return value!";
        case plssvm::target_platform::cpu:
            {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
                const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::cpu, 0);
                EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::cpu);
#else
                EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::cpu, 0), plssvm::hardware_sampling_exception, "Hardware sampling on CPUs wasn't enabled!");
#endif
            }
            break;
        case plssvm::target_platform::gpu_nvidia:
            {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
                const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_nvidia, 0);
                EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_nvidia);
#else
                EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_nvidia, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_nvidia' as target_platform, but hardware sampling on NVIDIA GPUs using NVML wasn't enabled! Try setting an nvidia target during CMake configuration.");
#endif
            }
            break;
        case plssvm::target_platform::gpu_amd:
            {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
                const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_amd, 0);
                EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_amd);
#else
                EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_amd, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_amd' as target_platform, but hardware sampling on AMD GPUs using ROCm SMI wasn't enabled! Try setting an amd target during CMake configuration.");
#endif
            }
            break;
        case plssvm::target_platform::gpu_intel:
            {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
                const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_intel, 0);
                EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_intel);
#else
                EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_intel, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_intel' as target_platform, but hardware sampling on Intel GPUs using Level Zero wasn't enabled! Try setting an intel target during CMake configuration.");
#endif
            }
            break;
    }
}

TEST(HardwareSamplerFactory, make_hardware_sampler_cpu) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::cpu, 0);
    EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::cpu);
#else
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::cpu, 0), plssvm::hardware_sampling_exception, "Hardware sampling on CPUs wasn't enabled!");
#endif
}

TEST(HardwareSamplerFactory, make_hardware_sampler_gpu_nvidia) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_NVIDIA_GPUS_ENABLED)
    const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_nvidia, 0);
    EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_nvidia);
#else
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_nvidia, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_nvidia' as target_platform, but hardware sampling on NVIDIA GPUs using NVML wasn't enabled! Try setting an nvidia target during CMake configuration.");
#endif
}

TEST(HardwareSamplerFactory, make_hardware_sampler_gpu_amd) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_AMD_GPUS_ENABLED)
    const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_amd, 0);
    EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_amd);
#else
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_amd, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_amd' as target_platform, but hardware sampling on AMD GPUs using ROCm SMI wasn't enabled! Try setting an amd target during CMake configuration.");
#endif
}

TEST(HardwareSamplerFactory, make_hardware_sampler_gpu_intel) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_INTEL_GPUS_ENABLED)
    const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> sampler = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_intel, 0);
    EXPECT_EQ(sampler->sampling_target(), plssvm::target_platform::gpu_intel);
#else
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::make_hardware_sampler(plssvm::target_platform::gpu_intel, 0), plssvm::hardware_sampling_exception, "Provided 'gpu_intel' as target_platform, but hardware sampling on Intel GPUs using Level Zero wasn't enabled! Try setting an intel target during CMake configuration.");
#endif
}

TEST(HardwareSamplerFactory, create_hardware_sampler_cpu_only_once) {
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
    // if CPU sampling is enabled, only one entry must be present
    const std::vector<std::unique_ptr<plssvm::detail::tracking::hardware_sampler>> sampler = plssvm::detail::tracking::create_hardware_sampler(plssvm::target_platform::cpu, 1);
    ASSERT_EQ(sampler.size(), 1);
    EXPECT_EQ(sampler.front()->sampling_target(), plssvm::target_platform::cpu);
#else
    // if CPU sampling is not enabled, an empty vector should be returned
    EXPECT_TRUE(plssvm::detail::tracking::create_hardware_sampler(plssvm::target_platform::cpu, 1).empty());
#endif
}

TEST(HardwareSamplerFactory, create_hardware_sampler_no_device) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::detail::tracking::create_hardware_sampler(plssvm::target_platform::automatic, 0), plssvm::hardware_sampling_exception, "The number of devices must be greater than 0!");
}
