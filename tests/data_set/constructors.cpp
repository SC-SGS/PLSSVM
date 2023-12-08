/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the different data_set constructors.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"      // plssvm::file_format_type
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_NEAR, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, label_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_distinct_label, get_correct_data_file_labels, generate_specific_matrix, scale}

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{ContainsRegex, StartsWith}
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE; ASSERT_TRUE, FAIL, ::testing::{Test, StaticAssertTypeEq}

#include <cstddef>      // std::size_t
#include <tuple>        // std::get
#include <type_traits>  // std::is_integral_v
#include <vector>       // std::vector

template <typename T>
class DataSetConstructors : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    /**
     * @brief Return the correct data points according to the ARFF and LIBSVM template files.
     * @return the correct data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::soa_matrix<plssvm::real_type> &get_correct_template_file_data_points() const noexcept { return correct_template_file_data_points_; }

  private:
    /// The correct data points.
    plssvm::soa_matrix<plssvm::real_type> correct_template_file_data_points_{ { { plssvm::real_type{ -1.117827500607882 }, plssvm::real_type{ -2.9087188881250993 }, plssvm::real_type{ 0.66638344270039144 }, plssvm::real_type{ 1.0978832703949288 } },
                                                                                { plssvm::real_type{ -0.5282118298909262 }, plssvm::real_type{ -0.335880984968183973 }, plssvm::real_type{ 0.51687296029754564 }, plssvm::real_type{ 0.54604461446026 } },
                                                                                { plssvm::real_type{ 0.57650218263054642 }, plssvm::real_type{ 1.01405596624706053 }, plssvm::real_type{ 0.13009428079760464 }, plssvm::real_type{ 0.7261913886869387 } },
                                                                                { plssvm::real_type{ -0.20981208921241892 }, plssvm::real_type{ 0.60276937379453293 }, plssvm::real_type{ -0.13086851759108944 }, plssvm::real_type{ 0.10805254527169827 } },
                                                                                { plssvm::real_type{ 1.88494043717792 }, plssvm::real_type{ 1.00518564317278263 }, plssvm::real_type{ 0.298499933047586044 }, plssvm::real_type{ 1.6464627048813514 } },
                                                                                { plssvm::real_type{ -1.1256816275635 }, plssvm::real_type{ 2.12541534341344414 }, plssvm::real_type{ -0.165126576545454511 }, plssvm::real_type{ 2.5164553141200987 } } },
                                                                              plssvm::PADDING_SIZE,
                                                                              plssvm::PADDING_SIZE };
};
TYPED_TEST_SUITE(DataSetConstructors, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetConstructors, typedefs) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a data_set using an existing LIBSVM data set file
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<label_type, typename decltype(data)::label_type>();
    EXPECT_TRUE(std::is_integral_v<typename decltype(data)::size_type>);
}

TYPED_TEST(DataSetConstructors, construct_arff_from_file_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->filename.append(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_arff_from_file_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::data_set<label_type> data{ PLSSVM_TEST_PATH "/data/arff/3x2_without_label.arff" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_libsvm_from_file_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_libsvm_from_file_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::data_set<label_type> data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSetConstructors, construct_explicit_arff_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::arff };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_explicit_libsvm_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSetConstructors, construct_scaled_arff_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->filename.append(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSetConstructors, construct_scaled_libsvm_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSetConstructors, construct_scaled_explicit_arff_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::arff, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSetConstructors, construct_scaled_explicit_libsvm_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), correct_labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSetConstructors, construct_scaled_too_many_factors) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.1 } },
        factors_type{ 1, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.1 } },
        factors_type{ 2, plssvm::real_type{ 2.0 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.0 }, plssvm::real_type{ 3.1 } },
        factors_type{ 4, plssvm::real_type{ 4.0 }, plssvm::real_type{ 4.1 } }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "Need at most as much scaling factors as features in the data set are present (4), but 5 were given!");
}
TYPED_TEST(DataSetConstructors, construct_scaled_invalid_feature_index) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 4, plssvm::real_type{ 4.0 }, plssvm::real_type{ 4.1 } },
        factors_type{ 2, plssvm::real_type{ 2.0 }, plssvm::real_type{ 2.1 } }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "The maximum scaling feature index most not be greater than 3, but is 4!");
}
TYPED_TEST(DataSetConstructors, construct_scaled_duplicate_feature_index) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
    // create (invalid) scaling factors
    scaling_type scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
    scaling.scaling_factors = std::vector<factors_type>{
        factors_type{ 1, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.1 } },
        factors_type{ 2, plssvm::real_type{ 2.0 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.0 }, plssvm::real_type{ 3.1 } },
        factors_type{ 2, plssvm::real_type{ 2.0 }, plssvm::real_type{ 2.1 } }
    };

    // try creating a data set with invalid scaling factors
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ this->filename, scaling }),
                      plssvm::data_set_exception,
                      "Found more than one scaling factor for the feature index 2!");
}

TYPED_TEST(DataSetConstructors, construct_from_vector_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points.to_2D_vector() };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), correct_data_points);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_from_empty_vector) {
    using label_type = typename TestFixture::fixture_label_type;

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ std::vector<std::vector<plssvm::real_type>>{} }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}
TYPED_TEST(DataSetConstructors, construct_from_vector_with_differing_num_features) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = {
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.1 } },
        { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.1 }, plssvm::real_type{ 1.2 } }
    };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points }),
                      plssvm::data_set_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}
TYPED_TEST(DataSetConstructors, construct_from_vector_with_no_features) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = { {}, {} };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points }),
                      plssvm::data_set_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(DataSetConstructors, construct_from_vector_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(labels.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points.to_2D_vector(), labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), correct_data_points);
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetConstructors, construct_from_vector_mismatching_num_data_points_and_labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const auto correct_data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create data set
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ correct_data_points,
                                                     std::vector<label_type>{ labels } }),
                      plssvm::data_set_exception,
                      fmt::format("Number of labels ({}) must match the number of data points ({})!", labels.size(), correct_data_points.num_rows()));
}

TYPED_TEST(DataSetConstructors, construct_scaled_from_vector_without_label) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ data_points.to_2D_vector(), scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSetConstructors, construct_scaled_from_vector_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(labels.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points.to_2D_vector(), labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

template <typename T>
class DataSetMatrixConstructors : public DataSetConstructors<T> {
  protected:
    using typename DataSetConstructors<T>::fixture_label_type;
    static constexpr plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};
TYPED_TEST_SUITE(DataSetMatrixConstructors, util::label_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_without_label_no_padding) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_TRUE(data.data().is_padded());
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_without_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points });
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSetMatrixConstructors, construct_from_empty_matrix_no_padding) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ plssvm::matrix<plssvm::real_type, layout>{} }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}
TYPED_TEST(DataSetMatrixConstructors, construct_from_empty_matrix) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ plssvm::matrix<plssvm::real_type, layout>{ 0, 0, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_with_label_no_padding) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data_points, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_TRUE(data.data().is_padded());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}
TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_with_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_without_label_no_padding) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data_points_scaled, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_TRUE(data.data().is_padded());
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_without_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points_scaled });
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_with_label_no_padding) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data_points_scaled, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_TRUE(data.data().is_padded());
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}
TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_with_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create data set
    const plssvm::data_set<label_type> data{ correct_data_points, labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points_scaled });
    EXPECT_TRUE(data.has_labels());
    ASSERT_TRUE(data.labels().has_value());
    EXPECT_EQ(data.labels().value().get(), labels);
    ASSERT_TRUE(data.classes().has_value());
    EXPECT_EQ(data.classes().value(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    EXPECT_TRUE(data.scaling_factors().has_value());
    ASSERT_EQ(data.scaling_factors().value().get().scaling_factors.size(), scaling_factors.size());
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        auto factors = data.scaling_factors().value().get().scaling_factors[i];
        EXPECT_EQ(factors.feature, std::get<0>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.lower, std::get<1>(scaling_factors[i]));
        EXPECT_FLOATING_POINT_NEAR(factors.upper, std::get<2>(scaling_factors[i]));
    }
}

TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_without_label_wrong_padding_sizes) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    {
        // create data points
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, 0, plssvm::PADDING_SIZE);
        // create data set
        EXPECT_THROW_WHAT(plssvm::data_set<label_type>{ data_points }, plssvm::data_set_exception, fmt::format("Expected {} as row-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
    {
        // create data points
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, plssvm::PADDING_SIZE, 0);
        // create data set
        EXPECT_THROW_WHAT(plssvm::data_set<label_type>{ data_points }, plssvm::data_set_exception, fmt::format("Expected {} as column-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
}
TYPED_TEST(DataSetMatrixConstructors, construct_from_matrix_with_label_wrong_padding_sizes) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    {
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, 0, plssvm::PADDING_SIZE);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, labels }), plssvm::data_set_exception, fmt::format("Expected {} as row-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
    {
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, plssvm::PADDING_SIZE, 0);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, labels }), plssvm::data_set_exception, fmt::format("Expected {} as column-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
}
TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_without_label_wrong_padding_sizes) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    {  // create data points
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, 0, plssvm::PADDING_SIZE);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }), plssvm::data_set_exception, fmt::format("Expected {} as row-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
    {  // create data points
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(4, 4, plssvm::PADDING_SIZE, 0);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, scaling_type{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }), plssvm::data_set_exception, fmt::format("Expected {} as column-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
}
TYPED_TEST(DataSetMatrixConstructors, construct_scaled_from_matrix_with_label_wrong_padding_sizes) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    {
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, 0, plssvm::PADDING_SIZE);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, labels, { -1.0, 1.0 } }), plssvm::data_set_exception, fmt::format("Expected {} as row-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
    {
        const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(labels.size(), 4, plssvm::PADDING_SIZE, 0);
        // create data set
        EXPECT_THROW_WHAT((plssvm::data_set<label_type>{ data_points, labels, { -1.0, 1.0 } }), plssvm::data_set_exception, fmt::format("Expected {} as column-padding size, but got 0!", plssvm::PADDING_SIZE));
    }
}