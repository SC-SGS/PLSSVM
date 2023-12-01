/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the utility functions used for parsing a LIBSVM model file.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "naming.hpp"  // naming::{pretty_print_x_vs_y, pretty_print_calc_alpha_idx}

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_P, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_DEATH
                                   // ::testing::{TestWithParam, Values}

#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::make_tuple, std::ignore
#include <vector>   // std::vector

class LIBSVMModelUtilityXvsY : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> {};

TEST_P(LIBSVMModelUtilityXvsY, x_vs_y_to_idx) {
    const auto [x, y, num_classes, expected_idx] = GetParam();

    EXPECT_EQ(plssvm::detail::io::x_vs_y_to_idx(x, y, num_classes), expected_idx);
    EXPECT_EQ(plssvm::detail::io::x_vs_y_to_idx(y, x, num_classes), expected_idx);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(LIBSVMModelUtilityXvsY, LIBSVMModelUtilityXvsY, ::testing::Values(
        std::make_tuple(0, 1, 4, 0), std::make_tuple(0, 2, 4, 1), std::make_tuple(0, 3, 4, 2), std::make_tuple(1, 2, 4, 3),
        std::make_tuple(1, 3, 4, 4), std::make_tuple(2, 3, 4, 5), std::make_tuple(0, 1, 2, 0), std::make_tuple(1, 0, 2, 0),
        std::make_tuple(1, 2, 3, 2), std::make_tuple(2, 1, 3, 2)),
        naming::pretty_print_x_vs_y<LIBSVMModelUtilityXvsY>);
// clang-format on

TEST(LIBSVMModelUtilityXvsYDeathTest, x_equal_to_y) {
    EXPECT_DEATH(std::ignore = plssvm::detail::io::x_vs_y_to_idx(0, 0, 2), "Can't compute the index for the binary classification of 0vs0!");
}
TEST(LIBSVMModelUtilityXvsYDeathTest, too_few_classes) {
    EXPECT_DEATH(std::ignore = plssvm::detail::io::x_vs_y_to_idx(0, 1, 1), "There must be at least two classes!");
}
TEST(LIBSVMModelUtilityXvsYDeathTest, x_greater_or_equal_than_num_classes) {
    EXPECT_DEATH(std::ignore = plssvm::detail::io::x_vs_y_to_idx(3, 0, 2), ::testing::HasSubstr("The class x (3) must be smaller than the total number of classes (2)!"));
}
TEST(LIBSVMModelUtilityXvsYDeathTest, y_greater_or_equal_than_num_classes) {
    EXPECT_DEATH(std::ignore = plssvm::detail::io::x_vs_y_to_idx(0, 3, 3), ::testing::HasSubstr("The class y (3) must be smaller than the total number of classes (3)!"));
}

class LIBSVMModelUtilityAlphaIdx : public ::testing::TestWithParam<std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>> {
  protected:
    void SetUp() override {
        index_sets_ = std::vector<std::vector<std::size_t>>{
            { 0, 2, 4 },   // 0
            { 1, 3, 5 },   // 1
            { 6, 8, 10 },  // 2
            { 7, 9 }       // 3
        };
    }

    /**
     * @brief Return the indices of the support vectors used for testing the `plssvm::detail::io::calculate_alpha_idx` function.
     * @return the indices (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::vector<std::size_t>> &get_index_sets() const noexcept { return index_sets_; }

  private:
    /// The support vector index sets per class.
    std::vector<std::vector<std::size_t>> index_sets_{};
};

TEST_P(LIBSVMModelUtilityAlphaIdx, calculate_alpha_idx) {
    const auto [i, j, idx_to_find, expected_global_idx] = GetParam();

    EXPECT_EQ(plssvm::detail::io::calculate_alpha_idx(i, j, this->get_index_sets(), idx_to_find), expected_global_idx);
    EXPECT_EQ(plssvm::detail::io::calculate_alpha_idx(j, i, this->get_index_sets(), idx_to_find), expected_global_idx);
}

// clang-format off
INSTANTIATE_TEST_SUITE_P(LIBSVMModelUtilityAlphaIdx, LIBSVMModelUtilityAlphaIdx, ::testing::Values(
        std::make_tuple(0, 2, 4, 2), std::make_tuple(0, 2, 10, 5), std::make_tuple(0, 1, 1, 3), std::make_tuple(0, 1, 0, 0),
        std::make_tuple(0, 3, 7, 3), std::make_tuple(0, 3, 9, 4), std::make_tuple(3, 2, 8, 1), std::make_tuple(3, 2, 9, 4),
        std::make_tuple(1, 2, 3, 1), std::make_tuple(2, 1, 3, 1)),
        naming::pretty_print_calc_alpha_idx<LIBSVMModelUtilityAlphaIdx>);
// clang-format on

TEST(LIBSVMModelUtilityAlphaIdxDeathTest, i_equal_to_j) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 1 }, { 2, 3 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 0, index_sets, 0), "Can't compute the index for 0 == 0!");
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, too_few_index_sets) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 1 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 1, index_sets, 0), "At least two index sets must be provided!");
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, i_greater_or_equal_than_num_indices) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 1 }, { 2, 3 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(3, 0, index_sets, 0), ::testing::HasSubstr("The index i (3) must be smaller than the total number of indices (2)!"));
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, j_greater_or_equal_than_num_indices) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 1 }, { 2, 3 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 2, index_sets, 0), ::testing::HasSubstr("The index j (2) must be smaller than the total number of indices (2)!"));
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, index_sets_not_sorted) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 2, 1 }, { 3, 5, 4 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 1, index_sets, 0), "The index sets must be sorted in ascending order!");
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, indices_in_one_index_set_not_unique) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 0 }, { 2, 3 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 1, index_sets, 1), "All indices in one index set must be unique!");
}
TEST(LIBSVMModelUtilityAlphaIdxDeathTest, index_sets_not_disjoint) {
    const std::vector<std::vector<std::size_t>> index_sets{ { 0, 1 }, { 1, 3 } };
    EXPECT_DEATH(std::ignore = plssvm::detail::io::calculate_alpha_idx(0, 1, index_sets, 1), "The content of both index sets must be disjoint!");
}
