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

#include "gmock/gmock.h"  // EXPECT_THAT; ::testing::AnyOf
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::kokkos::execution_space -> std::string conversions are correct
TEST(KokkosExecutionSpace, to_string) {
    // check conversions to std::string
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
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::kokkos::execution_space>(9), "unknown");
}

// check whether the std::string -> plssvm::kokkos::execution_space conversions are correct
TEST(KokkosExecutionSpace, from_string) {
    // check conversion from std::string
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

TEST(KokkosExecutionSpace, execution_space_to_kokkos_type) {
    // check conversions
#if defined(KOKKOS_ENABLE_CUDA)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::cuda>, Kokkos::Cuda>();
#endif
#if defined(KOKKOS_ENABLE_HIP)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::hip>, Kokkos::HIP>();
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::sycl>, Kokkos::SYCL>();
#endif
#if defined(KOKKOS_ENABLE_HPX)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::hpx>, Kokkos::Experimental::HPX>();
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::openmp>, Kokkos::OpenMP>();
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::openmp_target>, Kokkos::Experimental::OpenMPTarget>();
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::openacc>, Kokkos::OpenACC>();
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::threads>, Kokkos::Threads>();
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::serial>, Kokkos::Serial>();
#endif
}

TEST(KokkosExecutionSpace, kokkos_type_to_execution_space) {
    // check conversions
#if defined(KOKKOS_ENABLE_CUDA)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Cuda>, plssvm::kokkos::execution_space::cuda);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::HIP>, plssvm::kokkos::execution_space::hip);
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::SYCL>, plssvm::kokkos::execution_space::sycl);
#endif
#if defined(KOKKOS_ENABLE_HPX)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Experimental::HPX>, plssvm::kokkos::execution_space::hpx);
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::OpenMP>, plssvm::kokkos::execution_space::openmp);
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Experimental::OpenMPTarget>, plssvm::kokkos::execution_space::openmp_target);
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Experimental::OpenACC>, plssvm::kokkos::execution_space::openacc);
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Threads>, plssvm::kokkos::execution_space::threads);
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::Serial>, plssvm::kokkos::execution_space::serial);
#endif
}

TEST(KokkosExecutionSpace, constexpr_available_execution_spaces) {
    // at least one execution space must always be available
    EXPECT_FALSE(plssvm::kokkos::detail::constexpr_available_execution_spaces().empty());
}

TEST(KokkosExecutionSpace, list_available_execution_spaces) {
    // at least one execution space must always be available
    EXPECT_FALSE(plssvm::kokkos::list_available_execution_spaces().empty());
}
