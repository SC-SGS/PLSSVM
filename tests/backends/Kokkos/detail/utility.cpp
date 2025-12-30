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

#include "plssvm/backends/execution_range.hpp"               // plssvm::detail::dim_type
#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"  // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_spaces.hpp"       // plssvm::kokkos::execution_space
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "tests/utility.hpp"  // util::for_each_variant_type

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_GE, SCOPED_TRACE

#include <map>      // std::map
#include <regex>    // std::regex, std::regex::extended, std::regex_match
#include <string>   // std::string
#include <variant>  // std::variant
#include <vector>   // std::vector

TEST(KokkosUtility, IsTypeInVariant) {
    // check type trait that determines if a type is contained in a type trait
    using variant_type = std::variant<int, double, bool, std::string>;

    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<int, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<double, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<bool, variant_type>) );
    EXPECT_TRUE((plssvm::kokkos::detail::impl::is_type_in_variant_v<std::string, variant_type>) );
    EXPECT_FALSE((plssvm::kokkos::detail::impl::is_type_in_variant_v<short, variant_type>) );
    EXPECT_FALSE((plssvm::kokkos::detail::impl::is_type_in_variant_v<float, variant_type>) );
}

TEST(KokkosUtility, DimTypeToNative) {
    // create a dim_type
    constexpr plssvm::detail::dim_type dim{ 128ull, 64ull, 32ull };

    // convert it to a Kokkos one-dimensional execution range
    const int native_dim = plssvm::kokkos::detail::dim_type_to_native(dim);

    // check values for correctness
    EXPECT_EQ(native_dim, 262'144);  // = 128 * 64 * 32
}

TEST(KokkosUtility, AvailableTargetPlatformToExecutionSpaceMapping) {
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
        // get the device name of the specified Kokkos execution space
        const std::string device_name = plssvm::kokkos::detail::get_device_name(plssvm::kokkos::detail::device_wrapper{ ExecutionSpace{} });
        SCOPED_TRACE(device_name);

        // must not be empty
        EXPECT_FALSE(device_name.empty());
        // must not start or end with whitespace
        const std::regex reg{ R"([^\s](?:.*[^\s])?)" };
        EXPECT_TRUE(std::regex_match(device_name, reg));
    }
};

TEST(KokkosUtility, GetDeviceName) {
    using variant_type = typename plssvm::kokkos::detail::impl::create_device_variant_type::type;
    util::for_each_variant_type<variant_type>(device_name_test{});
}

TEST(KokkosUtility, GetKokkosVersion) {
    const std::regex reg{ "[0-9]+\\.[0-9]+\\.[0-9]+", std::regex::extended };
    EXPECT_TRUE(std::regex_match(plssvm::kokkos::detail::get_kokkos_version(), reg));
}
