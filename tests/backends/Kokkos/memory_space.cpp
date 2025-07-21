/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different Kokkos execution spaces.
 */

#include "plssvm/backends/Kokkos/memory_space.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE

#include <sstream>  // std::istringstream

// check whether the plssvm::kokkos::memory_space -> std::string conversions are correct
TEST(KokkosMemorySpace, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::host_space, "HostSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::cuda_space, "CudaSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::cuda_usm_space, "CudaUVMSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::hip_space, "HIPSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::hip_usm_space, "HIPManagedSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::sycl_space, "SYCLDeviceUSMSpace");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::memory_space::sycl_usm_space, "SYCLSharedUSMSpace");
}

TEST(KokkosMemorySpace, to_string_unknown) {
    // check conversions to std::string from unknown memory_space
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::kokkos::memory_space>(7), "unknown");
}

// check whether the std::string -> plssvm::kokkos::memory_space conversions are correct
TEST(KokkosMemorySpace, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("HostSpace", plssvm::kokkos::memory_space::host_space);
    EXPECT_CONVERSION_FROM_STRING("host_space", plssvm::kokkos::memory_space::host_space);
    EXPECT_CONVERSION_FROM_STRING("CudaSpace", plssvm::kokkos::memory_space::cuda_space);
    EXPECT_CONVERSION_FROM_STRING("cuda_space", plssvm::kokkos::memory_space::cuda_space);
    EXPECT_CONVERSION_FROM_STRING("CudaUVMSpace", plssvm::kokkos::memory_space::cuda_usm_space);
    EXPECT_CONVERSION_FROM_STRING("cuda_usm_space", plssvm::kokkos::memory_space::cuda_usm_space);
    EXPECT_CONVERSION_FROM_STRING("HIPSpace", plssvm::kokkos::memory_space::hip_space);
    EXPECT_CONVERSION_FROM_STRING("hip_space", plssvm::kokkos::memory_space::hip_space);
    EXPECT_CONVERSION_FROM_STRING("HIPManagedSpace", plssvm::kokkos::memory_space::hip_usm_space);
    EXPECT_CONVERSION_FROM_STRING("hip_usm_space", plssvm::kokkos::memory_space::hip_usm_space);
    EXPECT_CONVERSION_FROM_STRING("SYCLDeviceUSMSpace", plssvm::kokkos::memory_space::sycl_space);
    EXPECT_CONVERSION_FROM_STRING("sycl_space", plssvm::kokkos::memory_space::sycl_space);
    EXPECT_CONVERSION_FROM_STRING("SYCLSharedUSMSpace", plssvm::kokkos::memory_space::sycl_usm_space);
    EXPECT_CONVERSION_FROM_STRING("sycl_usm_space", plssvm::kokkos::memory_space::sycl_usm_space);
}

TEST(KokkosMemorySpace, from_string_unknown) {
    // foo isn't a valid memory_space
    std::istringstream input{ "foo" };
    plssvm::kokkos::memory_space space{};
    input >> space;
    EXPECT_TRUE(input.fail());
}

TEST(KokkosMemorySpace, list_available_memory_spaces) {
    const std::vector<plssvm::kokkos::memory_space> memory_spaces = plssvm::kokkos::list_available_memory_spaces();

    // at least one must be available (host_space)!
    EXPECT_GE(memory_spaces.size(), 1);

    // the host memory space must always be present
    EXPECT_THAT(memory_spaces, ::testing::Contains(plssvm::kokkos::memory_space::host_space));
}
