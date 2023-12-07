/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the data_set getter member functions.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/matrix.hpp"     // plssvm::aos_matrix

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_EQ, EXPECT_FLOATING_POINT_NEAR
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"             // util::{redirect_output, scale}

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{ContainsRegex, StartsWith}
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EQ, ASSERT_TRUE, ::testing::Test

#include <cstddef>  // std::size_t
#include <tuple>    // std::get
#include <vector>   // std::vector

template <typename T>
class DataSetGetter : public ::testing::Test, private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    /**
     * @brief Return the different classes.
     * @return the classes (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<fixture_label_type> &get_classes() const noexcept { return classes_; }
    /**
     * @brief Return the correct labels.
     * @return the correct labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<fixture_label_type> &get_label() const noexcept { return label_; }
    /**
     * @brief Return the correct data points.
     * @return the correct data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::soa_matrix<plssvm::real_type> &get_data_points() const noexcept { return data_points_; }

  private:
    /// The correct, different classes.
    std::vector<fixture_label_type> classes_{ util::get_distinct_label<fixture_label_type>() };
    /// The correct labels.
    std::vector<fixture_label_type> label_{ util::get_correct_data_file_labels<fixture_label_type>() };
    /// The correct data points.
    plssvm::soa_matrix<plssvm::real_type> data_points_{ util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label_.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE) };
};
TYPED_TEST_SUITE(DataSetGetter, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetGetter, data) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // check data getter
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_data_points());
}
TYPED_TEST(DataSetGetter, has_labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_labels{ this->get_data_points() };
    // check has_labels getter
    EXPECT_FALSE(data_without_labels.has_labels());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->get_data_points(), this->get_label() };
    // check has_labels getter
    EXPECT_TRUE(data_with_labels.has_labels());
}
TYPED_TEST(DataSetGetter, labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_labels{ this->get_data_points() };
    // check labels getter
    EXPECT_FALSE(data_without_labels.labels().has_value());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->get_data_points(), this->get_label() };
    // check labels getter
    ASSERT_TRUE(data_with_labels.labels().has_value());
    EXPECT_EQ(data_with_labels.labels().value().get(), this->get_label());
}
TYPED_TEST(DataSetGetter, classes) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_labels{ this->get_data_points() };
    // check different_labels getter
    EXPECT_FALSE(data_without_labels.classes().has_value());
    // create data set with labels
    const plssvm::data_set<label_type> data_with_labels{ this->get_data_points(), this->get_label() };
    // check different_labels getter
    ASSERT_TRUE(data_with_labels.classes().has_value());
    EXPECT_EQ(data_with_labels.classes().value(), this->get_classes());
}

TYPED_TEST(DataSetGetter, num_data_points) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // check num_data_points getter
    EXPECT_EQ(data.num_data_points(), this->get_data_points().num_rows());
}
TYPED_TEST(DataSetGetter, num_features) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // check num_features getter
    EXPECT_EQ(data.num_features(), this->get_data_points().num_cols());
}
TYPED_TEST(DataSetGetter, num_classes) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data_without_label{ this->get_data_points() };
    // check num_different_labels getter
    EXPECT_EQ(data_without_label.num_classes(), 0);

    // create data set with labels
    const plssvm::data_set<label_type> data_with_label{ this->get_data_points(), this->get_label() };
    // check num_different_labels getter
    EXPECT_EQ(data_with_label.num_classes(), this->get_classes().size());
}

TYPED_TEST(DataSetGetter, is_scaled) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data set
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // check is_scaled getter
    EXPECT_FALSE(data.is_scaled());

    // create scaled data set
    const plssvm::data_set<label_type> data_scaled{ this->get_data_points(), scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };
    // check is_scaled getter
    EXPECT_TRUE(data_scaled.is_scaled());
}
TYPED_TEST(DataSetGetter, scaling_factors) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data set
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // check scaling_factors getter
    EXPECT_FALSE(data.scaling_factors().has_value());

    // create scaled data set
    const plssvm::data_set<label_type> data_scaled{ this->get_data_points(), scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };
    // check scaling_factors getter
    ASSERT_TRUE(data_scaled.scaling_factors().has_value());
    const auto &[ignored, correct_scaling_factors] = util::scale(this->get_data_points(), plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    const scaling_type &scaling_factors = data_scaled.scaling_factors().value();
    EXPECT_FLOATING_POINT_EQ(scaling_factors.scaling_interval.first, plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(scaling_factors.scaling_interval.second, plssvm::real_type{ 1.0 });
    ASSERT_EQ(scaling_factors.scaling_factors.size(), correct_scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.scaling_factors.size(); ++i) {
        EXPECT_EQ(scaling_factors.scaling_factors[i].feature, std::get<0>(correct_scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(scaling_factors.scaling_factors[i].lower, std::get<1>(correct_scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(scaling_factors.scaling_factors[i].upper, std::get<2>(correct_scaling_factors[i]));
    }
}