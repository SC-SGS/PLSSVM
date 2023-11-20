/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the fields in the version header.
 */

#include "plssvm/version/version.hpp"
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::is_populated

#include "naming.hpp"  // naming::pretty_print_version_info

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EQ, EXPECT_THAT, ASSERT_TRUE, ASSERT_FALSE,
                                   // ::testing::{TestWithParam, Combine, Values, Bool}

#include <optional>  // std::optional
#include <string>    // std::string
#include <tuple>     // std::tuple

TEST(Version, name_not_empty) {
    EXPECT_FALSE(plssvm::version::name.empty());
}
TEST(Version, version_not_empty) {
    EXPECT_FALSE(plssvm::version::version.empty());
}

TEST(Version, version_major_not_negative) {
    EXPECT_TRUE(plssvm::version::major >= 0);
}
TEST(Version, version_minor_not_negative) {
    EXPECT_TRUE(plssvm::version::minor >= 0);
}
TEST(Version, version_patch_not_negative) {
    EXPECT_TRUE(plssvm::version::patch >= 0);
}

TEST(Version, version_string_major_minor_patch) {
    EXPECT_EQ(plssvm::version::version, fmt::format("{}.{}.{}", plssvm::version::major, plssvm::version::minor, plssvm::version::patch));
}

TEST(Version, target_platforms_not_empty) {
    EXPECT_FALSE(plssvm::version::detail::target_platforms.empty());
}
TEST(Version, copyright_notice_not_empty) {
    EXPECT_FALSE(plssvm::version::detail::copyright_notice.empty());
}

TEST(Version, get_git_info_conditional_empty) {
    const std::optional<std::string> git_info = plssvm::version::detail::get_git_info();
    if (plssvm::version::git_metadata::is_populated()) {
        ASSERT_TRUE(git_info.has_value());
        if (git_info.has_value()) {
            EXPECT_FALSE(git_info->empty());
        }
    } else {
        EXPECT_FALSE(git_info.has_value());
    }
}

class VersionGetVersionInfo : public ::testing::TestWithParam<std::tuple<std::string, bool>> {};
TEST_P(VersionGetVersionInfo, get_version_info_not_empty) {
    const auto &[exe_name, with_backend_info] = GetParam();
    const std::string version_info = plssvm::version::detail::get_version_info(exe_name, with_backend_info);
    ASSERT_FALSE(version_info.empty());
    EXPECT_THAT(version_info, ::testing::HasSubstr(exe_name));
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(Version, VersionGetVersionInfo, ::testing::Combine(
                ::testing::Values("plssvm-train", "plssvm-predict", "plssvm-scale"),
                ::testing::Bool()),
                naming::pretty_print_version_info<VersionGetVersionInfo>);
// clang-format on