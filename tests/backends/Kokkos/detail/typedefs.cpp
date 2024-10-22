/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the Kokkos::View typedefs.
 */

#include "plssvm/backends/Kokkos/detail/typedefs.hpp"  // plssvm::kokkos::detail::{device_view_type, host_view_type}

#include "Kokkos_Core.hpp"  // Kokkos::View, Kokkos::DefaultExecutionSpace, Kokkos::HostSpace, Kokkos::MemoryUnmanaged

#include "gtest/gtest.h"  // TEST, ::testing::StaticAssertTypeEq

TEST(KokkosTypedefs, device_view_type) {
    // test device view typedefs
    ::testing::StaticAssertTypeEq<Kokkos::View<int *, Kokkos::DefaultExecutionSpace>, plssvm::kokkos::detail::device_view_type<int>>();
    ::testing::StaticAssertTypeEq<Kokkos::View<const unsigned *, Kokkos::DefaultExecutionSpace>, plssvm::kokkos::detail::device_view_type<const unsigned>>();
}

TEST(KokkosTypedefs, host_view_type) {
    // test host view typedefs
    ::testing::StaticAssertTypeEq<Kokkos::View<double *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>, plssvm::kokkos::detail::host_view_type<double>>();
    ::testing::StaticAssertTypeEq<Kokkos::View<const float *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>, plssvm::kokkos::detail::host_view_type<const float>>();
}
