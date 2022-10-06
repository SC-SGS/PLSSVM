/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the layout implementation.
 */

#include "plssvm/detail/layout.hpp"

#include "../naming.hpp"  // util::{arithmetic_types_to_name}
#include "utility.hpp"    // util::{convert_to_string, convert_from_string, redirect_output}

#include "fmt/format.h"   // fmt::format, fmt::join
#include "gtest/gtest.h"  // TEST, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ::testing::{Test, Types}

#include <cstddef>  // std::size_t
#include <sstream>  // std::istringstream
#include <tuple>    // std::ignore
#include <vector>   // std::vector

// check whether the plssvm::detail::layout_type -> std::string conversions are correct
TEST(Layout, to_string) {
    // check conversion to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::detail::layout_type::aos), "Array-of-Structs (AoS)");
    EXPECT_EQ(util::convert_to_string(plssvm::detail::layout_type::soa), "Struct-of-Arrays (SoA)");
}
TEST(Layout, to_string_unknown) {
    // check conversions to std::string from unknown layout_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::detail::layout_type>(2)), "unknown");
}

// check whether the std::string -> plssvm::detail::layout_type conversions are correct
TEST(Layout, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::detail::layout_type>("aos"), plssvm::detail::layout_type::aos);
    EXPECT_EQ(util::convert_from_string<plssvm::detail::layout_type>("Array-of-Structs"), plssvm::detail::layout_type::aos);
    EXPECT_EQ(util::convert_from_string<plssvm::detail::layout_type>("soa"), plssvm::detail::layout_type::soa);
    EXPECT_EQ(util::convert_from_string<plssvm::detail::layout_type>("Struct-of-Arrays"), plssvm::detail::layout_type::soa);
}
TEST(Layout, from_string_unknown) {
    // foo isn't a valid layout_type
    std::istringstream input{ "foo" };
    plssvm::detail::layout_type layout;
    input >> layout;
    EXPECT_TRUE(input.fail());
}

template <typename T>
class Layout : public ::testing::Test, private util::redirect_output {
  protected:
    void SetUp() override {
        // create matrix
        matrix = {
            { real_type{ 11 }, real_type{ 12 }, real_type{ 13 }, real_type{ 14 }, real_type{ 15 } },
            { real_type{ 21 }, real_type{ 22 }, real_type{ 23 }, real_type{ 24 }, real_type{ 25 } },
            { real_type{ 31 }, real_type{ 32 }, real_type{ 33 }, real_type{ 34 }, real_type{ 35 } },
            { real_type{ 41 }, real_type{ 42 }, real_type{ 43 }, real_type{ 44 }, real_type{ 45 } }
        };
        num_points = matrix.size();
        num_features = matrix.front().size();
    }

    using real_type = T;

    std::vector<std::vector<real_type>> matrix{};
    std::size_t num_points{};
    std::size_t num_features{};
};

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Layout, floating_point_types, naming::arithmetic_types_to_name);

TYPED_TEST(Layout, array_of_structs) {
    // correctly transformed matrix in AoS layout without boundary
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 0, this->num_points, this->num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 0, this->num_points);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));
}
TYPED_TEST(Layout, array_of_structs_boundary) {
    // correctly transformed matrix in AoS layout with a boundary of size 2
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0, 41, 42, 43, 44, 45, 0, 0 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 2, this->num_points, this->num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 2, this->num_points);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));
}
TYPED_TEST(Layout, array_of_structs_fewer_data_points) {
    // correctly transformed matrix in AoS layout with fewer data points and without boundary
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 2, 3, this->num_features);
    EXPECT_EQ(aos_direct, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_direct, ", "), fmt::join(correct_aos, ", "));

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 2, 3);
    EXPECT_EQ(aos_indirect, correct_aos) << fmt::format("result: [{}], correct: [{}]", fmt::join(aos_indirect, ", "), fmt::join(correct_aos, ", "));
}

TYPED_TEST(Layout, struct_of_arrays) {
    // correctly transformed matrix in SoA layout without boundary
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43, 14, 24, 34, 44, 15, 25, 35, 45 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 0, this->num_points, this->num_features);
    EXPECT_EQ(soa_direct, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_direct, ", "), fmt::join(correct_soa, ", "));

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 0, this->num_points);
    EXPECT_EQ(soa_indirect, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_indirect, ", "), fmt::join(correct_soa, ", "));
}
TYPED_TEST(Layout, struct_of_arrays_boundary) {
    // correctly transformed matrix in SoA layout with a boundary of size 2
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 41, 0, 0, 12, 22, 32, 42, 0, 0, 13, 23, 33, 43, 0, 0, 14, 24, 34, 44, 0, 0, 15, 25, 35, 45, 0, 0 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 2, this->num_points, this->num_features);
    EXPECT_EQ(soa_direct, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_direct, ", "), fmt::join(correct_soa, ", "));

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 2, this->num_points);
    EXPECT_EQ(soa_indirect, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_indirect, ", "), fmt::join(correct_soa, ", "));
}
TYPED_TEST(Layout, struct_of_arrays_fewer_data_points) {
    // correctly transformed matrix in SoA layout with fewer data points and without boundary
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 0, 0, 12, 22, 32, 0, 0, 13, 23, 33, 0, 0, 14, 24, 34, 0, 0, 15, 25, 35, 0, 0 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 2, 3, this->num_features);
    EXPECT_EQ(soa_direct, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_direct, ", "), fmt::join(correct_soa, ", "));

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 2, 3);
    EXPECT_EQ(soa_indirect, correct_soa) << fmt::format("result: [{}], correct: [{}]", fmt::join(soa_indirect, ", "), fmt::join(correct_soa, ", "));
}

template <typename T>
class LayoutDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(LayoutDeathTest, floating_point_types, naming::arithmetic_types_to_name);

TYPED_TEST(LayoutDeathTest, empty_matrix) {
    // matrix must not be empty
    const std::vector<std::vector<TypeParam>> matrix{};
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 0), "Matrix is empty!");
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 0, 0), "Matrix is empty!");
}

TYPED_TEST(LayoutDeathTest, too_many_data_points_to_transform) {
    // number of points to transform must be smaller than total number of data points
    const std::vector<std::vector<TypeParam>> matrix = {
        { 11, 12, 13 },
        { 21, 22, 23 }
    };
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 3),
                 "Number of data points to transform can not exceed matrix size!");
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 0, 3),
                 "Number of data points to transform can not exceed matrix size!");
}

TYPED_TEST(LayoutDeathTest, different_number_of_features) {
    // number of features must be the same for all data points
    const std::vector<std::vector<TypeParam>> matrix = {
        { 11, 12, 13 },
        { 21, 22 }
    };
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 2),
                 "Feature sizes mismatch! All features should have size 3.");
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 0, 2),
                 "Feature sizes mismatch! All features should have size 3.");
}

TYPED_TEST(LayoutDeathTest, empty_features) {
    // number of features must be the same for all data points
    const std::vector<std::vector<TypeParam>> matrix = { {}, {} };
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, matrix, 0, 2), "All features are empty!");
    EXPECT_DEATH(std::ignore = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, matrix, 0, 2), "All features are empty!");
}