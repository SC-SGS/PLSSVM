/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the different classification data set constructors.
 */

#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"  // data set class to test
#include "plssvm/data_set/min_max_scaler.hpp"           // plssvm::min_max_scaler
#include "plssvm/exceptions/exceptions.hpp"             // plssvm::data_set_exception, plssvm::mpi_exception
#include "plssvm/file_format_types.hpp"                 // plssvm::file_format_type
#include "plssvm/matrix.hpp"                            // plssvm::matrix, plssvm::layout_type
#include "plssvm/shape.hpp"                             // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_NEAR, EXPECT_THROW_WHAT, EXPECT_OPTIONAL_EQ
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{classification_label_type_gtest, classification_label_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_distinct_label, get_correct_data_file_labels, generate_specific_matrix, scale}

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator

    #include "mpi.h"  // MPI_COMM_WORLD, MPI_Comm_dup, MPI_Comm_free
#endif

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE; ASSERT_TRUE, FAIL, ::testing::{Test, StaticAssertTypeEq}

#include <cstddef>      // std::size_t
#include <tuple>        // std::get
#include <type_traits>  // std::is_integral_v
#include <utility>      // std::move
#include <vector>       // std::vector

template <typename T>
class ClassificationDataSetConstructors : public ::testing::Test,
                                          private util::redirect_output<>,
                                          protected util::temporary_file {
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
                                                                                { plssvm::real_type{ -1.1256816275635 }, plssvm::real_type{ 2.12541534341344414 }, plssvm::real_type{ -0.165126576545454511 }, plssvm::real_type{ 2.5164553141200987 } } } };
};

TYPED_TEST_SUITE(ClassificationDataSetConstructors, util::classification_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ClassificationDataSetConstructors, Typedefs) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a data_set using an existing LIBSVM data set file
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<label_type, typename decltype(data)::label_type>();
    EXPECT_TRUE(std::is_integral_v<typename decltype(data)::size_type>);
}

//*************************************************************************************************************************************//
//                                                         construct from file                                                         //
//*************************************************************************************************************************************//

TYPED_TEST(ClassificationDataSetConstructors, ConstructARFFFromFileWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->append_to_filename(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructARFFFromFileWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::classification_data_set<label_type> data{ PLSSVM_TEST_PATH "/data/arff/3x2_without_label.arff" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data }));
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructLIBSVMFromFileWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructLIBSVMFromFileWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    const plssvm::classification_data_set<label_type> data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // check values
    const std::vector<std::vector<plssvm::real_type>> correct_data = {
        { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
        { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), (plssvm::soa_matrix<plssvm::real_type>{ correct_data }));
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), 3);
    EXPECT_EQ(data.num_features(), 2);
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructExplicitARFFFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, plssvm::file_format_type::arff };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructExplicitLIBSVMFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), this->get_correct_template_file_data_points());
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledARFFFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->append_to_filename(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledARFFFromFileCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // must append .arff to filename so that the correct function is called in the data_set constructor
    this->append_to_filename(".arff");

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, this->filename, { comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledLIBSVMFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledLIBSVMFromFileCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, this->filename, { comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledExplicitARFFFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, plssvm::file_format_type::arff, { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledExplicitARFFFromFileCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/arff/classification/6x4_TEMPLATE.arff", this->filename);
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, this->filename, plssvm::file_format_type::arff, { comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledExplicitLIBSVMFromFile) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    const plssvm::classification_data_set<label_type> data{ this->filename, plssvm::file_format_type::libsvm, { plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } } };

    const std::vector<label_type> correct_different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> correct_labels = util::get_correct_data_file_labels<label_type>();

    // check values
    const auto [scaled_data_points, scaling_factors] = util::scale(this->get_correct_template_file_data_points(), plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ scaled_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), correct_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), correct_different_labels);

    EXPECT_EQ(data.num_data_points(), this->get_correct_template_file_data_points().num_rows());
    EXPECT_EQ(data.num_features(), this->get_correct_template_file_data_points().num_cols());
    EXPECT_EQ(data.num_classes(), correct_different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledExplicitLIBSVMFromFileCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, this->filename, plssvm::file_format_type::libsvm, { comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

//*************************************************************************************************************************************//
//                                                      construct from 2D vector                                                       //
//*************************************************************************************************************************************//

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromVectorWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points.to_2D_vector() };

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

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromEmptyVector) {
    using label_type = typename TestFixture::fixture_label_type;

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ std::vector<std::vector<plssvm::real_type>>{} }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromVectorWithDifferingNumFeatures) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = {
        { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.1 } },
        { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.1 }, plssvm::real_type{ 1.2 } }
    };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ correct_data_points }),
                      plssvm::data_set_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromVectorWithNoFeatures) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const std::vector<std::vector<plssvm::real_type>> correct_data_points = { {}, {} };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ correct_data_points }),
                      plssvm::data_set_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromVectorWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points.to_2D_vector(), labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), correct_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromEmptyVectorAndLabels) {
    using label_type = typename TestFixture::fixture_label_type;

    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ std::vector<std::vector<plssvm::real_type>>{}, labels }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructFromVectorMismatchingNumDataPointsAndLabels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const auto correct_data_points = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create data set
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ correct_data_points,
                                                                    std::vector<label_type>{ labels } }),
                      plssvm::data_set_exception,
                      fmt::format("Number of labels ({}) must match the number of data points ({})!", labels.size(), correct_data_points.num_rows()));
}

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledFromVectorWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ data_points.to_2D_vector(), plssvm::min_max_scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

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
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledFromVectorWithoutLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, data_points.to_2D_vector(), plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledFromVectorWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points.to_2D_vector(), labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetConstructors, ConstructScaledFromVectorWithLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, correct_data_points.to_2D_vector(), labels, plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

//*************************************************************************************************************************************//
//                                                        construct from matrix                                                        //
//*************************************************************************************************************************************//

template <typename T>
class ClassificationDataSetMatrixConstructors : public ClassificationDataSetConstructors<T> {
  protected:
    using typename ClassificationDataSetConstructors<T>::fixture_label_type;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};

TYPED_TEST_SUITE(ClassificationDataSetMatrixConstructors, util::classification_label_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructFromMatrixWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ 4, 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points };

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

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructFromEmptyMatrix) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    const plssvm::matrix<plssvm::real_type, layout> data_points{ plssvm::shape{ 0, 0 } };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ data_points }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructFromMatrixWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ labels.size(), 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points, labels };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructFromEmptyMatrixWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const plssvm::matrix<plssvm::real_type, layout> data_points{ plssvm::shape{ 0, 0 } };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ data_points, labels }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructFromMatrixWithLabelSizeMismatch) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const plssvm::matrix<plssvm::real_type, layout> data_points{ plssvm::shape{ labels.size() - 1, 4 } };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT_MATCHER((plssvm::classification_data_set<label_type>{ data_points, labels }),
                              plssvm::data_set_exception,
                              ::testing::HasSubstr(fmt::format("Number of labels ({}) must match the number of data points ({})!", labels.size(), labels.size() - 1)));
}

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructScaledFromMatrixWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ 4, 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ data_points, plssvm::min_max_scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

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
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructScaledFromMatrixWithoutLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points
    const auto data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ 4, 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, data_points, plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructScaledFromMatrixWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ labels.size(), 4 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ correct_data_points, labels, { -1.0, 1.0 } };

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });
    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), plssvm::soa_matrix<plssvm::real_type>{ correct_data_points_scaled });
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetMatrixConstructors, ConstructScaledFromMatrixWithLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto correct_data_points = util::generate_specific_matrix<plssvm::matrix<plssvm::real_type, layout>>(plssvm::shape{ labels.size(), 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, correct_data_points, labels, plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

//*************************************************************************************************************************************//
//                                                    construct from r-value matrix                                                    //
//*************************************************************************************************************************************//

template <typename T>
class ClassificationDataSetRValueMatrixConstructors : public ClassificationDataSetConstructors<T> {
  protected:
    using typename ClassificationDataSetConstructors<T>::fixture_label_type;
};

TYPED_TEST_SUITE(ClassificationDataSetRValueMatrixConstructors, util::classification_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructFromRvalueMatrixWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto copied_correct_data_points = correct_data_points;

    // create data set
    const plssvm::classification_data_set<label_type> data{ std::move(correct_data_points) };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), copied_correct_data_points);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), copied_correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), copied_correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructFromEmptyRvalueMatrix) {
    using label_type = typename TestFixture::fixture_label_type;

    plssvm::soa_matrix<plssvm::real_type> data_points{ plssvm::shape{ 0, 0 } };

    // creating a data set from an empty vector is illegal
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ std::move(data_points) }),
                      plssvm::data_set_exception,
                      "Data vector is empty!");
}

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructFromRvalueMatrixWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const std::vector<label_type> copied_labels = labels;
    auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });
    const auto copied_correct_data_points = correct_data_points;

    // create data set
    const plssvm::classification_data_set<label_type> data{ std::move(correct_data_points), std::move(labels) };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_EQ(data.data(), copied_correct_data_points);
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), copied_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), copied_correct_data_points.num_rows());
    EXPECT_EQ(data.num_features(), copied_correct_data_points.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_FALSE(data.is_scaled());
    EXPECT_FALSE(data.scaling_factors().has_value());
}

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructScaledFromRvalueMatrixWithoutLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points
    auto data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ std::move(data_points), plssvm::min_max_scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_FALSE(data.has_labels());
    EXPECT_FALSE(data.labels().has_value());
    EXPECT_FALSE(data.classes().has_value());

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), 0);

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructScaledFromRvalueMatrixWithoutLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points
    auto data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, std::move(data_points), plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructScaledFromRvalueMatrixWithLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const std::vector<label_type> copied_labels = labels;
    auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });

    const auto [correct_data_points_scaled, scaling_factors] = util::scale(correct_data_points, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });

    // create data set
    const plssvm::classification_data_set<label_type> data{ std::move(correct_data_points), std::move(labels), { -1.0, 1.0 } };

    // check values
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data.data(), correct_data_points_scaled);
    EXPECT_TRUE(data.has_labels());
    EXPECT_OPTIONAL_EQ(data.labels(), copied_labels);
    EXPECT_OPTIONAL_EQ(data.classes(), different_labels);

    EXPECT_EQ(data.num_data_points(), correct_data_points_scaled.num_rows());
    EXPECT_EQ(data.num_features(), correct_data_points_scaled.num_cols());
    EXPECT_EQ(data.num_classes(), different_labels.size());

    EXPECT_TRUE(data.is_scaled());
    const auto &data_scaling_factors_opt = data.scaling_factors();
    EXPECT_TRUE(data_scaling_factors_opt.has_value());
    if (data_scaling_factors_opt.has_value()) {
        const auto &scaling_factors_opt = data_scaling_factors_opt.value().get().scaling_factors();
        ASSERT_TRUE(scaling_factors_opt.has_value());
        if (scaling_factors_opt.has_value()) {
            const std::vector<plssvm::min_max_scaler::factors> factors = scaling_factors_opt.value();
            ASSERT_EQ(factors.size(), scaling_factors.size());
            for (std::size_t i = 0; i < factors.size(); ++i) {
                EXPECT_EQ(factors[i].feature, std::get<0>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].lower, std::get<1>(scaling_factors[i]));
                EXPECT_FLOATING_POINT_NEAR(factors[i].upper, std::get<2>(scaling_factors[i]));
            }
        }
    }
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(ClassificationDataSetRValueMatrixConstructors, ConstructScaledFromRvalueMatrixWithLabelCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data points and labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();
    std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    auto correct_data_points = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 4 });
    EXPECT_THROW_WHAT((plssvm::classification_data_set<label_type>{ plssvm::mpi::communicator{}, std::move(correct_data_points), std::move(labels), plssvm::min_max_scaler{ comm, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } }),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the data set and scaler must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif
