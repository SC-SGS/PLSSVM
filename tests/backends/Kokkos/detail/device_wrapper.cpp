/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the device_wrapper class.
 */

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"

#include "plssvm/backends/Kokkos/detail/utility.hpp"   // plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::{execution_space, kokkos_type_to_execution_space_v}
#include "plssvm/detail/utility.hpp"                   // plssvm::detail::contains
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace

#include "tests/utility.hpp"  // util::for_each_variant_type

#include "gtest/gtest.h"  // TEST, EXPECT_GE, EXPECT_EQ

#include <vector>  // std::vector

TEST(KokkosDeviceWrapper, default_construct) {
    // default construct a device wrapper
    const plssvm::kokkos::detail::device_wrapper device{};

    // per std::variant specification, the first type in the underlying variant is now the active member
    // -> this always corresponds to the first entry in our constexpr_available_execution_spaces array
    constexpr auto spaces = plssvm::kokkos::detail::constexpr_available_execution_spaces();
    EXPECT_EQ(device.get_execution_space(), spaces.front());
}

TEST(KokkosDeviceWrapper, construct) {
    // construct a device wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // check that the device is associated with the correct execution space
    EXPECT_EQ(device.get_execution_space(), plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>);
}

TEST(KokkosDeviceWrapper, get) {
    // construct a device wrapper using the current Kokkos::DefaultExecutionSpace
    plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // check that the returned Kokkos::ExecutionSpace has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(device.get<space>()), Kokkos::DefaultExecutionSpace &>();
}

TEST(KokkosDeviceWrapper, get_const) {
    // construct a device wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // check that the returned Kokkos::ExecutionSpace has the correct type
    constexpr plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>;
    ::testing::StaticAssertTypeEq<decltype(device.get<space>()), const Kokkos::DefaultExecutionSpace &>();
}

TEST(KokkosDeviceWrapper, get_execution_space) {
    // construct a device wrapper using the current Kokkos::DefaultExecutionSpace
    const plssvm::kokkos::detail::device_wrapper device{ Kokkos::DefaultExecutionSpace{} };

    // check that the device is associated with the correct execution space
    EXPECT_EQ(device.get_execution_space(), plssvm::kokkos::kokkos_type_to_execution_space_v<Kokkos::DefaultExecutionSpace>);
}

TEST(KokkosDeviceWrapper, equality) {
    const plssvm::kokkos::detail::device_wrapper device1{ Kokkos::DefaultExecutionSpace{} };
    const plssvm::kokkos::detail::device_wrapper device2{ Kokkos::DefaultExecutionSpace{} };

    // should be equal
    EXPECT_TRUE(device1 == device2);
}

TEST(KokkosDeviceWrapper, inequality) {
    const plssvm::kokkos::detail::device_wrapper device1{ Kokkos::DefaultExecutionSpace{} };
    const plssvm::kokkos::detail::device_wrapper device2{ Kokkos::DefaultExecutionSpace{} };

    // should not be unequal
    EXPECT_FALSE(device1 != device2);
}

struct device_list_test {
    template <typename ExecutionSpace>
    void operator()() const {
        // get the default device list
        const plssvm::kokkos::execution_space space = plssvm::kokkos::kokkos_type_to_execution_space_v<ExecutionSpace>;
        plssvm::target_platform default_target{};
        for (const auto &[target, spaces] : plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping()) {
            if (::plssvm::detail::contains(spaces, space)) {
                default_target = target;
                break;
            }
        }
        const std::vector<plssvm::kokkos::detail::device_wrapper> devices = plssvm::kokkos::detail::get_device_list(space, default_target);

        // check the number of returned devices
        if (space == plssvm::kokkos::execution_space::cuda || space == plssvm::kokkos::execution_space::hip || space == plssvm::kokkos::execution_space::sycl) {
            // TODO: Change if multi-GPU support for Kokkos::Experimental::OpenMPTarget and/or Kokkos::Experimental::OpenACC is implemented
            // for the device execution spaces AT LEAST ONE device must be found
            EXPECT_GE(devices.size(), 1);
        } else {
            // for all other execution spaces EXACTLY ONE device must be found
            EXPECT_EQ(devices.size(), 1);
        }
    }
};

TEST(KokkosDeviceWrapper, get_device_list) {
    using variant_type = typename plssvm::kokkos::detail::impl::create_device_variant_type::type;
    util::for_each_variant_type<variant_type>(device_list_test{});
}
