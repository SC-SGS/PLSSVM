/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different Kokkos execution spaces.
 */

#include "plssvm/backends/Kokkos/memory_space_type_traits.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, ::testing::StaticAssertTypeEq

TEST(KokkosMemorySpaceTypeTraits, memory_space_to_kokkos_type) {
    // check conversions
#if defined(KOKKOS_ENABLE_CUDA)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::cuda_space>, Kokkos::CudaSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::cuda_usm_space>, Kokkos::CudaUVMSpace>();
#endif
#if defined(KOKKOS_ENABLE_HIP)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::hip_space>, Kokkos::HIPSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::hip_usm_space>, Kokkos::HIPManagedSpace>();
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::sycl_space>, Kokkos::SYCLDeviceUSMSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::sycl_usm_space>, Kokkos::SYCLSharedUSMSpace>();
#endif
    ::testing::StaticAssertTypeEq<plssvm::kokkos::memory_space_to_kokkos_type_t<plssvm::kokkos::memory_space::host_space>, Kokkos::HostSpace>();
}

TEST(KokkosMemorySpaceTypeTraits, kokkos_type_to_memory_space) {
    // check conversions
#if defined(KOKKOS_ENABLE_CUDA)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::CudaSpace>, plssvm::kokkos::memory_space::cuda_space);
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::CudaUVMSpace>, plssvm::kokkos::memory_space::cuda_usm_space);
#endif
#if defined(KOKKOS_ENABLE_HIP)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::HIPSpace>, plssvm::kokkos::memory_space::hip_space);
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::HIPManagedSpace>, plssvm::kokkos::memory_space::hip_usm_space);
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::SYCLDeviceUSMSpace>, plssvm::kokkos::memory_space::sycl_space);
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::SYCLSharedUSMSpace>, plssvm::kokkos::memory_space::sycl_usm_space);
#endif
    EXPECT_EQ(plssvm::kokkos::kokkos_type_to_memory_space_v<Kokkos::HostSpace>, plssvm::kokkos::memory_space::host_space);
}

TEST(KokkosMemorySpaceTypeTraits, execution_space_to_memory_space) {
    // check conversion
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::cuda, false>), plssvm::kokkos::memory_space::cuda_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::cuda, true>), plssvm::kokkos::memory_space::cuda_usm_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::hip, false>), plssvm::kokkos::memory_space::hip_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::hip, true>), plssvm::kokkos::memory_space::hip_usm_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::sycl, false>), plssvm::kokkos::memory_space::sycl_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::sycl, true>), plssvm::kokkos::memory_space::sycl_usm_space);

    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::hpx, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::hpx, true>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openmp, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openmp, true>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openmp_target, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openmp_target, true>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openacc, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::openacc, true>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::threads, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::threads, true>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::serial, false>), plssvm::kokkos::memory_space::host_space);
    EXPECT_EQ((plssvm::kokkos::execution_space_to_memory_space_v<plssvm::kokkos::execution_space::serial, true>), plssvm::kokkos::memory_space::host_space);
}

TEST(KokkosMemorySpaceTypeTraits, kokkos_execution_space_to_kokkos_memory_space) {
    // check conversions
#if defined(KOKKOS_ENABLE_CUDA)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Cuda, false>, Kokkos::CudaSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Cuda, true>, Kokkos::CudaUVMSpace>();
#endif
#if defined(KOKKOS_ENABLE_HIP)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::HIP, false>, Kokkos::HIPSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::HIP, true>, Kokkos::HIPManagedSpace>();
#endif
#if defined(KOKKOS_ENABLE_SYCL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::SYCL, false>, Kokkos::SYCLDeviceUSMSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::SYCL, true>, Kokkos::SYCLSharedUSMSpace>();
#endif
#if defined(KOKKOS_ENABLE_HPX)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::HPX, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::HPX, true>, Kokkos::HostSpace>();
#endif
#if defined(KOKKOS_ENABLE_OPENMP)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::OpenMP, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::OpenMP, true>, Kokkos::HostSpace>();
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::OpenMPTarget, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::OpenMPTarget, true>, Kokkos::HostSpace>();
#endif
#if defined(KOKKOS_ENABLE_OPENACC)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::OpenACC, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Experimental::OpenACC, true>, Kokkos::HostSpace>();
#endif
#if defined(KOKKOS_ENABLE_THREADS)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Threads, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Threads, true>, Kokkos::HostSpace>();
#endif
#if defined(KOKKOS_ENABLE_SERIAL)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Serial, false>, Kokkos::HostSpace>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::Serial, true>, Kokkos::HostSpace>();
#endif
}
