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

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"  // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/exceptions.hpp"             // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"        // plssvm::kokkos::{execution_space, kokkos_type_to_execution_space_v}
#include "plssvm/detail/utility.hpp"                         // plssvm::detail::contains
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "Kokkos_Core.hpp"  // Kokkos::ExecutionSpace

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::for_each_variant_type

#include "fmt/core.h"     // fmt::format
#include "gmock/gmock.h"  // EXPECT_THAT; ::testing::AnyOf
#include "gtest/gtest.h"  // TEST, EXPECT_NE

#include <map>      // std::map
#include <regex>    // std::regex, std::regex::extended, std::regex_match
#include <string>   // std::string
#include <variant>  // std::variant
#include <vector>   // std::vector

TEST(KokkosUtility, is_type_in_variant) {
    // check type trait that determines if a type is contained in a type trait
    using variant_type = std::variant<int, double, bool, std::string>;

    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<int, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<double, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<bool, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<std::string, variant_type>) );
    EXPECT_FALSE((plssvm::kokkos::detail::impl::is_type_in_variant_v<short, variant_type>) );
    EXPECT_FALSE((plssvm::kokkos::detail::impl::is_type_in_variant_v<float, variant_type>) );
}

TEST(KokkosUtility, available_target_platform_to_execution_space_mapping) {
    // get the target_platform <-> execution_space mappings
    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> mapping = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();

    // the map must not be empty
    EXPECT_FALSE(mapping.empty());

    // each vector must at least have one entry + the automatic target platform must not be present
    for (const auto &[target, spaces] : mapping) {
        EXPECT_NE(target, plssvm::target_platform::automatic);
        EXPECT_GE(spaces.size(), 1);
    }
}

struct device_name_test {
    template <typename ExecutionSpace>
    void operator()() const {
        // get the device name of the default Kokkos execution space
        const std::string name = plssvm::kokkos::detail::get_device_name(plssvm::kokkos::detail::device_wrapper{ ExecutionSpace{} });
        SCOPED_TRACE(name);

        // the returned device name may not be empty or unknown
        EXPECT_FALSE(name.empty());
        EXPECT_NE(name, std::string{ "unknown" });
    }
};

TEST(KokkosUtility, get_device_name) {
    using variant_type = typename plssvm::kokkos::detail::impl::create_device_variant_type::type;
    util::for_each_variant_type<variant_type>(device_name_test{});
}

TEST(KokkosUtility, get_kokkos_version) {
    const std::regex reg{ "[0-9]+\\.[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::kokkos::detail::get_kokkos_version(), reg));
}
