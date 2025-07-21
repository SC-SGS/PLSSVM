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

#include "plssvm/backends/Kokkos/detail/constexpr_available_memory_spaces.hpp"  // plssvm::kokkos::detail::constexpr_available_memory_spaces
#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"                     // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"                           // plssvm::kokkos::memory_space
#include "plssvm/backends/Kokkos/execution_space_type_traits.hpp"               // plssvm::kokkos::kokkos_type_to_execution_space_v
#include "plssvm/backends/Kokkos/memory_space_type_traits.hpp"                  // plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace, Kokkos::View

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <cstddef>  // std::size_t

TEST(KokkosDeviceViewWrapper, default_construct) {
    // default construct a device view wrapper
    const plssvm::kokkos::detail::device_view_wrapper<double *> view{};

    // per std::variant specification, the first type in the underlying variant is now the active member
    // -> this always corresponds to the first entry in our constexpr_available_memory_spaces array
    constexpr auto spaces = plssvm::kokkos::detail::constexpr_available_memory_spaces();
    EXPECT_EQ(view.get_memory_space(), spaces.front());
}

TEST(KokkosDeviceViewWrapper, construct) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = false;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, kokkos_memory_space>{}, use_usm_allocations };

    // check that the device view is associated with the correct memory space
    EXPECT_EQ(view.get_memory_space(), plssvm::kokkos::kokkos_type_to_memory_space_v<kokkos_memory_space>);
}

TEST(KokkosDeviceViewWrapper, get) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = false;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, kokkos_memory_space>{}, use_usm_allocations };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), Kokkos::View<double *, kokkos_memory_space> &>();
    ::testing::StaticAssertTypeEq<decltype(view.get<space>()), Kokkos::View<double *, kokkos_memory_space> &>();
}

TEST(KokkosDeviceViewWrapper, get_const) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = false;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<int **, kokkos_memory_space>{}, use_usm_allocations };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), const Kokkos::View<int **, kokkos_memory_space> &>();
    ::testing::StaticAssertTypeEq<decltype(view.get<space>()), const Kokkos::View<int **, kokkos_memory_space> &>();
}

TEST(KokkosDeviceViewWrapper, get_memory_space) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace>;
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, kokkos_memory_space>{} };

    // check that the device view is associated with the correct memory space
    EXPECT_EQ(view.get_memory_space(), plssvm::kokkos::kokkos_type_to_memory_space_v<kokkos_memory_space>);
}

TEST(KokkosDeviceViewWrapper, equality) {
    // get the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace>;

    const plssvm::kokkos::detail::device_view_wrapper view1{ Kokkos::View<double *, kokkos_memory_space>{} };
    const plssvm::kokkos::detail::device_view_wrapper view2{ Kokkos::View<double *, kokkos_memory_space>{} };

    // should be equal
    EXPECT_TRUE(view1 == view2);
}

TEST(KokkosDeviceViewWrapper, inequality) {
    // get the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace>;

    const plssvm::kokkos::detail::device_view_wrapper view1{ Kokkos::View<double *, kokkos_memory_space>{} };
    const plssvm::kokkos::detail::device_view_wrapper view2{ Kokkos::View<double *, kokkos_memory_space>{} };

    // should not be unequal
    EXPECT_FALSE(view1 != view2);
}

TEST(KokkosDeviceViewWrapper, make_device_view_wrapper) {
    // get the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = false;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;

    // create a device wrapper for the Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // create device view wrapper
    const plssvm::kokkos::detail::device_view_wrapper<double *> view = plssvm::kokkos::detail::make_device_view_wrapper<double *>(device, 42, use_usm_allocations);

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), const Kokkos::View<double *, kokkos_memory_space> &>();

    // check the number of elements
    EXPECT_EQ(view.get<space>().size(), std::size_t{ 42 });
}

TEST(KokkosUSMDeviceViewWrapper, construct) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = true;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, kokkos_memory_space>{}, use_usm_allocations };

    // check that the device view is associated with the correct memory space
    EXPECT_EQ(view.get_memory_space(), plssvm::kokkos::kokkos_type_to_memory_space_v<kokkos_memory_space>);
}

TEST(KokkosUSMDeviceViewWrapper, get) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = true;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<double *, kokkos_memory_space>{}, use_usm_allocations };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), Kokkos::View<double *, kokkos_memory_space> &>();
}

TEST(KokkosUSMDeviceViewWrapper, get_const) {
    // construct a device view wrapper using the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = true;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;
    const plssvm::kokkos::detail::device_view_wrapper view{ Kokkos::View<int **, kokkos_memory_space>{}, use_usm_allocations };

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), const Kokkos::View<int **, kokkos_memory_space> &>();
}

TEST(KokkosUSMDeviceViewWrapper, make_device_view_wrapper) {
    // get the Kokkos::MemorySpace associated with the current Kokkos::DefaultExecutionSpace
    constexpr bool use_usm_allocations = true;
    using kokkos_memory_space = plssvm::kokkos::kokkos_execution_space_to_kokkos_memory_space_t<Kokkos::DefaultExecutionSpace, use_usm_allocations>;

    // create a device wrapper for the Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // create device view wrapper
    const plssvm::kokkos::detail::device_view_wrapper<double *> view = plssvm::kokkos::detail::make_device_view_wrapper<double *>(device, 42, use_usm_allocations);

    // check that the returned Kokkos::View has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(view.get<space, use_usm_allocations>()), const Kokkos::View<double *, kokkos_memory_space> &>();

    // check the number of elements
    EXPECT_EQ((view.get<space, use_usm_allocations>().size()), std::size_t{ 42 });
}
