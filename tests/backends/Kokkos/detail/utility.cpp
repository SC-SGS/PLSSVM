/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom utility functions related to the Kokkos backend.
 */

#include "plssvm/backends/Kokkos/detail/utility.hpp"

#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT

#include "fmt/core.h"     // fmt::format
#include "gmock/gmock.h"  // EXPECT_THAT; ::testing::AnyOf
#include "gtest/gtest.h"  // TEST, EXPECT_GE, EXPECT_NE

#include <regex>   // std::regex, std::regex::extended, std::regex_match
#include <string>  // std::string
#include <vector>  // std::vector

TEST(KokkosUtility, determine_default_target_platform_from_execution_space) {
    // determine the potential default target platform
    EXPECT_EQ(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::cuda), plssvm::target_platform::gpu_nvidia);
    EXPECT_THAT(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::hip), ::testing::AnyOf(plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd));
    EXPECT_NE(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::sycl), plssvm::target_platform::automatic);
    EXPECT_EQ(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::hpx), plssvm::target_platform::cpu);
    EXPECT_EQ(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::openmp), plssvm::target_platform::cpu);
    EXPECT_NE(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::openmp_target), plssvm::target_platform::automatic);
    EXPECT_NE(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::openacc), plssvm::target_platform::automatic);
    EXPECT_EQ(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::threads), plssvm::target_platform::cpu);
    EXPECT_EQ(plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(plssvm::kokkos::execution_space::serial), plssvm::target_platform::cpu);
}

TEST(KokkosUtility, check_execution_space_target_platform_combination) {
    // check some execution_space <-> target_platform combinations
    // the cuda execution space only supports the NVIDIA GPU
    EXPECT_NO_THROW(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::cuda, plssvm::target_platform::gpu_nvidia));
    EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::cuda, plssvm::target_platform::gpu_amd),
                      plssvm::kokkos::backend_exception,
                      "The target platform gpu_amd is not supported for Kokkos Cuda execution space!");
    EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::cuda, plssvm::target_platform::gpu_intel),
                      plssvm::kokkos::backend_exception,
                      "The target platform gpu_intel is not supported for Kokkos Cuda execution space!");
    EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::cuda, plssvm::target_platform::cpu),
                      plssvm::kokkos::backend_exception,
                      "The target platform cpu is not supported for Kokkos Cuda execution space!");

    // the hip execution space only supports the NVIDIA and AMD GPUs
    EXPECT_NO_THROW(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::hip, plssvm::target_platform::gpu_nvidia));
    EXPECT_NO_THROW(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::hip, plssvm::target_platform::gpu_amd));
    EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::hip, plssvm::target_platform::gpu_intel),
                      plssvm::kokkos::backend_exception,
                      "The target platform gpu_intel is not supported for Kokkos HIP execution space!");
    EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(plssvm::kokkos::execution_space::hip, plssvm::target_platform::cpu),
                      plssvm::kokkos::backend_exception,
                      "The target platform cpu is not supported for Kokkos HIP execution space!");

    // TODO: SYCL
    // TODO: OpenMP target
    // TODO: OpenACC

    // the remaining execution spaces all only support CPUs!
    for (const plssvm::kokkos::execution_space exec : { plssvm::kokkos::execution_space::hpx, plssvm::kokkos::execution_space::openmp, plssvm::kokkos::execution_space::threads, plssvm::kokkos::execution_space::serial }) {
        EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(exec, plssvm::target_platform::gpu_nvidia),
                          plssvm::kokkos::backend_exception,
                          fmt::format("The target platform gpu_nvidia is not supported for Kokkos {} execution space!", exec));
        EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(exec, plssvm::target_platform::gpu_amd),
                          plssvm::kokkos::backend_exception,
                          fmt::format("The target platform gpu_amd is not supported for Kokkos {} execution space!", exec));
        EXPECT_THROW_WHAT(plssvm::kokkos::detail::check_execution_space_target_platform_combination(exec, plssvm::target_platform::gpu_intel),
                          plssvm::kokkos::backend_exception,
                          fmt::format("The target platform gpu_intel is not supported for Kokkos {} execution space!", exec));
        EXPECT_NO_THROW(plssvm::kokkos::detail::check_execution_space_target_platform_combination(exec, plssvm::target_platform::cpu));
    }
}

TEST(KokkosUtility, get_device_list) {
    // get the default device list
    const plssvm::kokkos::execution_space space = plssvm::kokkos::determine_default_execution_space();
    const plssvm::target_platform target = plssvm::kokkos::detail::determine_default_target_platform_from_execution_space(space);
    const std::vector<Kokkos::DefaultExecutionSpace> devices = plssvm::kokkos::detail::get_device_list(space, target);

    // check the number of returned devices
    if (space == plssvm::kokkos::execution_space::cuda || space == plssvm::kokkos::execution_space::hip || space == plssvm::kokkos::execution_space::sycl) {
        // for the device execution spaces AT LEAST ONE device must be found
        EXPECT_GE(devices.size(), 1);
    } else {
        // for all other execution spaces EXACTLY ONE device must be found
        EXPECT_EQ(devices.size(), 1);
    }
}

TEST(KokkosUtility, get_device_name) {
    // get the device name of the default Kokkos execution space
    const plssvm::kokkos::execution_space space = plssvm::kokkos::determine_default_execution_space();
    const std::string name = plssvm::kokkos::detail::get_device_name(space, Kokkos::DefaultExecutionSpace{});

    // the returned device name may not be empty or unknown
    EXPECT_FALSE(name.empty());
    EXPECT_NE(name, std::string{ "unknown" });
}

TEST(KokkosUtility, get_kokkos_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::kokkos::detail::get_kokkos_version(), reg));
}
