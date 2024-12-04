/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the device_view_wrapper class.
 */

#include "plssvm/backends/Kokkos/detail/device_view_wrapper.hpp"

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"  // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"        // plssvm::kokkos::{execution_space, kokkos_type_to_execution_space_v}

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace, Kokkos::View

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <cstddef>  // std::size_t

TEST(KokkosDeviceViewWrapper, default_construct) {
    // default construct a device view wrapper
    const plssvm::kokkos::detail::device_view_wrapper<double *> view{};

    // per std::variant specification, the first type in the underlying variant is now the active member
    // -> this always corresponds to the first entry in our constexpr_available_execution_spaces array
    constexpr auto spaces = plssvm::kokkos::detail::constexpr_available_execution_spaces();
    EXPECT_EQ(view.get_execution_space(), spaces.front());
}

TEST(KokkosDeviceViewWrapper, construct) {
    // construct a device view wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };

    // check that the device view is associated with the correct execution space
    EXPECT_EQ(view.get_execution_space(), plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>);
}

TEST(KokkosDeviceViewWrapper, get) {
    // construct a device view wrapper using the current Kokkos::DefaultExecutionSpace
    plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space>()), Kokkos::View<double *, Kokkos::DefaultExecutionSpace> &>();
}

TEST(KokkosDeviceViewWrapper, get_const) {
    // construct a device view wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<int **, Kokkos::DefaultExecutionSpace>{} };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space>()), const Kokkos::View<int **, Kokkos::DefaultExecutionSpace> &>();
}

TEST(KokkosDeviceViewWrapper, get_execution_space) {
    // construct a device wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };

    // check that the device view is associated with the correct execution space
    EXPECT_EQ(view.get_execution_space(), plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>);
}

TEST(KokkosDeviceViewWrapper, equality) {
    const plssvm::kokkos::detail::device_view_wrapper view1{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };
    const plssvm::kokkos::detail::device_view_wrapper view2{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };

    // should be equal
    EXPECT_TRUE(view1 == view2);
}

TEST(KokkosDeviceViewWrapper, inequality) {
    const plssvm::kokkos::detail::device_view_wrapper view1{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };
    const plssvm::kokkos::detail::device_view_wrapper view2{ Kokkos::View<double *, Kokkos::DefaultExecutionSpace>{} };

    // should not be unequal
    EXPECT_FALSE(view1 != view2);
}

TEST(KokkosDeviceViewWrapper, make_device_view_wrapper) {
    // create a device wrapper for the Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // create device view wrapper
    const plssvm::kokkos::detail::device_view_wrapper<double *> view = plssvm::kokkos::detail::make_device_view_wrapper<double *>(device, 42);

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space>()), const Kokkos::View<double *, Kokkos::DefaultExecutionSpace> &>();

    // check the number of elements
    EXPECT_EQ(view.get<space>().size(), std::size_t{ 42 });
}
