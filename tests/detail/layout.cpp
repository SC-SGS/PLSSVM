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

#include "../custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_FLOATING_POINT_EQ, EXPECT_FLOATING_POINT_VECTOR_EQ, EXPECT_FLOATING_POINT_2D_VECTOR_EQ
#include "../naming.hpp"              // util::real_type_to_name
#include "../types_to_test.hpp"       // util::real_type_gtest
#include "../utility.hpp"             // util::redirect_output

#include "fmt/format.h"   // fmt::format, fmt::join
#include "gtest/gtest.h"  // TEST, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ::testing::{Test, Types}

#include <cstddef>  // std::size_t
#include <sstream>  // std::istringstream
#include <tuple>    // std::ignore
#include <vector>   // std::vector

// check whether the plssvm::detail::layout_type -> std::string conversions are correct
TEST(Layout, to_string) {
    // check conversion to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::layout_type::aos, "Array-of-Structs (AoS)");
    EXPECT_CONVERSION_TO_STRING(plssvm::detail::layout_type::soa, "Struct-of-Arrays (SoA)");
}
TEST(Layout, to_string_unknown) {
    // check conversions to std::string from unknown layout_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::detail::layout_type>(2), "unknown");
}

// check whether the std::string -> plssvm::detail::layout_type conversions are correct
TEST(Layout, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("aos", plssvm::detail::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("Array-of-Structs", plssvm::detail::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("soa", plssvm::detail::layout_type::soa);
    EXPECT_CONVERSION_FROM_STRING("Struct-of-Arrays", plssvm::detail::layout_type::soa);
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
TYPED_TEST_SUITE(Layout, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(Layout, array_of_structs) {
    // correctly transformed matrix in AoS layout without boundary
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 41, 42, 43, 44, 45 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 0, this->num_points, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_direct, correct_aos);

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 0, this->num_points);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_indirect, correct_aos);
}
TYPED_TEST(Layout, array_of_structs_boundary) {
    // correctly transformed matrix in AoS layout with a boundary of size 2
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0, 41, 42, 43, 44, 45, 0, 0 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 2, this->num_points, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_direct, correct_aos);

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 2, this->num_points);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_indirect, correct_aos);
}
TYPED_TEST(Layout, array_of_structs_fewer_data_points) {
    // correctly transformed matrix in AoS layout with fewer data points and without boundary
    const std::vector<typename TestFixture::real_type> correct_aos = { 11, 12, 13, 14, 15, 0, 0, 21, 22, 23, 24, 25, 0, 0, 31, 32, 33, 34, 35, 0, 0 };

    // convert to AoS using the direct function call
    const std::vector<typename TestFixture::real_type> aos_direct = plssvm::detail::transform_to_aos_layout(this->matrix, 2, 3, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_direct, correct_aos);

    // convert to AoS using the indirect function call
    const std::vector<typename TestFixture::real_type> aos_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::aos, this->matrix, 2, 3);
    EXPECT_FLOATING_POINT_VECTOR_EQ(aos_indirect, correct_aos);
}

TYPED_TEST(Layout, struct_of_arrays) {
    // correctly transformed matrix in SoA layout without boundary
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43, 14, 24, 34, 44, 15, 25, 35, 45 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 0, this->num_points, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_direct, correct_soa);

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 0, this->num_points);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_indirect, correct_soa);
}
TYPED_TEST(Layout, struct_of_arrays_boundary) {
    // correctly transformed matrix in SoA layout with a boundary of size 2
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 41, 0, 0, 12, 22, 32, 42, 0, 0, 13, 23, 33, 43, 0, 0, 14, 24, 34, 44, 0, 0, 15, 25, 35, 45, 0, 0 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 2, this->num_points, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_direct, correct_soa);

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 2, this->num_points);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_indirect, correct_soa);
}
TYPED_TEST(Layout, struct_of_arrays_fewer_data_points) {
    // correctly transformed matrix in SoA layout with fewer data points and without boundary
    const std::vector<typename TestFixture::real_type> correct_soa = { 11, 21, 31, 0, 0, 12, 22, 32, 0, 0, 13, 23, 33, 0, 0, 14, 24, 34, 0, 0, 15, 25, 35, 0, 0 };

    // convert to SoA using the direct function call
    const std::vector<typename TestFixture::real_type> soa_direct = plssvm::detail::transform_to_soa_layout(this->matrix, 2, 3, this->num_features);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_direct, correct_soa);

    // convert to SoA using the indirect function call
    const std::vector<typename TestFixture::real_type> soa_indirect = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, this->matrix, 2, 3);
    EXPECT_FLOATING_POINT_VECTOR_EQ(soa_indirect, correct_soa);
}

template <typename T>
class LayoutDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(LayoutDeathTest, util::real_type_gtest, naming::real_type_to_name);

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