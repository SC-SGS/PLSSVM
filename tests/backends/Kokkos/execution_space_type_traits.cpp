/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different Kokkos execution spaces.
 */

#include "plssvm/backends/Kokkos/execution_space_type_traits.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, ::testing::StaticAssertTypeEq

TEST(KokkosExecutionSpaceTypeTraits, execution_space_to_kokkos_type) {
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
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::openacc>, Kokkos::Experimental::OpenACC>();
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::threads>, Kokkos::Threads>();
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::execution_space_to_kokkos_type_t<plssvm::kokkos::execution_space::serial>, Kokkos::Serial>();
#endif
}

TEST(KokkosExecutionSpaceTypeTraits, kokkos_type_to_execution_space) {
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
