/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for the layout implementation.
*/

#include "plssvm/constants.hpp"
#include "plssvm/detail/layout.hpp"

#include "fmt/format.h"
#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ

#include <string>  // std::string
#include <vector>  // std::vector

TEST(BaseLayout, layout_output) {
    std::stringstream ss;

    // Array-of-Structs (AoS)
    ss << plssvm::detail::layout_type::aos;
    EXPECT_EQ(ss.str(), "Array-of-Structs (AoS)");
    std::stringstream{}.swap(ss);

    // Struct-of-Arrays (SoA)
    ss << plssvm::detail::layout_type::soa;
    EXPECT_EQ(ss.str(), "Struct-of-Arrays (SoA)");
    std::stringstream{}.swap(ss);

    // unknown
    ss << static_cast<plssvm::detail::layout_type>(2);
    EXPECT_EQ(ss.str(), "unknown");
    std::stringstream{}.swap(ss);
}


// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class BaseLayout : public ::testing::Test {};
TYPED_TEST_SUITE(BaseLayout, floating_point_types);

TYPED_TEST(BaseLayout, layout_aos) {
    using real_type = TypeParam;

    // save current verbose flag and disable verbosity
    const bool verbose_save = plssvm::verbose;
    plssvm::verbose = false;

    // define matrix to test
    const std::vector<std::vector<real_type>> matrix = {
        { 11, 12, 13, 14, 15 },
        { 21, 22, 23, 24, 25 },
        { 31, 32, 33, 34, 35 },
        { 41, 42, 43, 44, 45 }
    };
    const std::size_t num_points = matrix.size();
    const std::size_t num_features = matrix.front().size();

    // correctly transformed matrix in AoS layout without boundary
    std::vector<real_type> correct_aos = { 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45 };

    // convert to AoS using the direct function call
    std::vector<real_type> aos_direct = plssvm::detail::transform_to_aos_layout(matrix, 0, num_points, num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    std::vector<real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, num_points);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));


    // correctly transformed matrix in AoS layout with a boundary of size 2
    correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0, 41, 42, 43, 44, 45, 0, 0 };

    // convert to AoS using the direct function call
    aos_direct = plssvm::detail::transform_to_aos_layout(matrix, 2, num_points, num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 2, num_points);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));


    // correctly transformed matrix in AoS layout with fewer data points and without boundary
    correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0 };

    // convert to AoS using the direct function call
    aos_direct = plssvm::detail::transform_to_aos_layout(matrix, 2, 3, num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 2, 3);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));

    // restore verbosity
    plssvm::verbose = verbose_save;
}

TYPED_TEST(BaseLayout, layout_soa) {
    using real_type = TypeParam;

    // save current verbose flag and disable verbosity
    const bool verbose_save = plssvm::verbose;
    plssvm::verbose = false;

    // define matrix to test
    const std::vector<std::vector<real_type>> matrix = {
        { 11, 12, 13, 14, 15 },
        { 21, 22, 23, 24, 25 },
        { 31, 32, 33, 34, 35 },
        { 41, 42, 43, 44, 45 }
    };
    const std::size_t num_points = matrix.size();
    const std::size_t num_features = matrix.front().size();

    // correctly transformed matrix in SoA layout without boundary
    std::vector<real_type> correct_soa = { 11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43, 14, 24, 34, 44, 15, 25, 35, 45 };

    // convert to SoA using the direct function call
    std::vector<real_type> soa_direct = plssvm::detail::transform_to_soa_layout(matrix, 0, num_points, num_features);
    EXPECT_EQ(soa_direct, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_direct, ", "), fmt::join(correct_soa, ", "));

    // convert to SoA using the indirect function call
    std::vector<real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 0, num_points);
    EXPECT_EQ(soa_indirect, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_indirect, ", "), fmt::join(correct_soa, ", "));


    // correctly transformed matrix in SoA layout with a boundary of size 2
    correct_soa = { 11, 21, 31, 0, 0, 12, 22, 32, 0, 0, 13, 23, 33, 0, 0, 14, 24, 34, 0, 0, 15, 25, 35, 0, 0 };

    // convert to SoA using the direct function call
    soa_direct = plssvm::detail::transform_to_soa_layout(matrix, 2, 3, num_features);
    EXPECT_EQ(soa_direct, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_direct, ", "), fmt::join(correct_soa, ", "));

    // convert to SoA using the indirect function call
    soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 2, 3);
    EXPECT_EQ(soa_indirect, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_indirect, ", "), fmt::join(correct_soa, ", "));

    // restore verbosity
    plssvm::verbose = verbose_save;
}


#if defined(PLSSVM_ASSERT_ENABLED)

template <typename T>
class BaseLayoutDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(BaseLayoutDeathTest, floating_point_types);

TYPED_TEST(BaseLayoutDeathTest, layout) {
    using real_type = TypeParam;

    [[maybe_unused]] std::vector<real_type> res;

    // matrix must not be empty
    std::vector<std::vector<real_type>> matrix{};
    EXPECT_DEATH(res = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 0), "");

    // number of points to transform must be smaller than total number of data points
    matrix = {
        { 11, 12, 13 },
        { 21, 22, 23 }
    };
    EXPECT_DEATH(res = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 3), "");

    // number of features must be the same for all data points
    matrix = {
        { 11, 12, 13 },
        { 21, 22 }
    };
    EXPECT_DEATH(res = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 2), "");
}

#endif