/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different Kokkos execution spaces.
 */

#include "plssvm/backends/Kokkos/execution_space.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_FALSE

#include <sstream>  // std::istringstream

// check whether the plssvm::kokkos::execution_space -> std::string conversions are correct
TEST(KokkosExecutionSpace, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::cuda, "Cuda");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::hip, "HIP");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::sycl, "SYCL");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::hpx, "HPX");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::openmp, "OpenMP");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::openmp_target, "OpenMPTarget");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::openacc, "OpenACC");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::threads, "Threads");
    EXPECT_CONVERSION_TO_STRING(plssvm::kokkos::execution_space::serial, "Serial");
}

TEST(KokkosExecutionSpace, to_string_unknown) {
    // check conversions to std::string from unknown execution_space
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::kokkos::execution_space>(10), "unknown");
}

// check whether the std::string -> plssvm::kokkos::execution_space conversions are correct
TEST(KokkosExecutionSpace, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("Automatic", plssvm::kokkos::execution_space::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTO", plssvm::kokkos::execution_space::automatic);
    EXPECT_CONVERSION_FROM_STRING("Cuda", plssvm::kokkos::execution_space::cuda);
    EXPECT_CONVERSION_FROM_STRING("CUDA", plssvm::kokkos::execution_space::cuda);
    EXPECT_CONVERSION_FROM_STRING("Hip", plssvm::kokkos::execution_space::hip);
    EXPECT_CONVERSION_FROM_STRING("HIP", plssvm::kokkos::execution_space::hip);
    EXPECT_CONVERSION_FROM_STRING("Sycl", plssvm::kokkos::execution_space::sycl);
    EXPECT_CONVERSION_FROM_STRING("SYCL", plssvm::kokkos::execution_space::sycl);
    EXPECT_CONVERSION_FROM_STRING("Hpx", plssvm::kokkos::execution_space::hpx);
    EXPECT_CONVERSION_FROM_STRING("HPX", plssvm::kokkos::execution_space::hpx);
    EXPECT_CONVERSION_FROM_STRING("OpenMP", plssvm::kokkos::execution_space::openmp);
    EXPECT_CONVERSION_FROM_STRING("OPENMP", plssvm::kokkos::execution_space::openmp);
    EXPECT_CONVERSION_FROM_STRING("OpenMP_Target", plssvm::kokkos::execution_space::openmp_target);
    EXPECT_CONVERSION_FROM_STRING("OPENMPTARGET", plssvm::kokkos::execution_space::openmp_target);
    EXPECT_CONVERSION_FROM_STRING("OpenACC", plssvm::kokkos::execution_space::openacc);
    EXPECT_CONVERSION_FROM_STRING("OPENACC", plssvm::kokkos::execution_space::openacc);
    EXPECT_CONVERSION_FROM_STRING("threads", plssvm::kokkos::execution_space::threads);
    EXPECT_CONVERSION_FROM_STRING("THREADS", plssvm::kokkos::execution_space::threads);
    EXPECT_CONVERSION_FROM_STRING("std::threads", plssvm::kokkos::execution_space::threads);
    EXPECT_CONVERSION_FROM_STRING("Serial", plssvm::kokkos::execution_space::serial);
    EXPECT_CONVERSION_FROM_STRING("SERIAL", plssvm::kokkos::execution_space::serial);
}

TEST(KokkosExecutionSpace, from_string_unknown) {
    // foo isn't a valid execution_space
    std::istringstream input{ "foo" };
    plssvm::kokkos::execution_space space{};
    input >> space;
    EXPECT_TRUE(input.fail());
}

TEST(KokkosExecutionSpace, list_available_execution_spaces) {
    const std::vector<plssvm::kokkos::execution_space> execution_spaces = plssvm::kokkos::list_available_execution_spaces();

    // at least one must be available (automatic)!
    EXPECT_GE(execution_spaces.size(), 1);

    // the automatic execution space must always be present
    EXPECT_THAT(execution_spaces, ::testing::Contains(plssvm::kokkos::execution_space::automatic));
}
