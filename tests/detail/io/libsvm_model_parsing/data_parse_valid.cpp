/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing a valid LIBSVM model file data section.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/classification_types.hpp"   // plssvm::classification_type
#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR
#include "naming.hpp"              // naming::parameter_definition_to_name
#include "types_to_test.hpp"       // util::label_type_classification_type_gtest
#include "utility.hpp"             // util::{temporary_file, get_correct_model_file_labels, get_distinct_label, generate_specific_matrix}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, ASSERT_EQ, FAIL, ::testing::Test

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::ignore
#include <vector>   // std::vector

template <typename T>
class LIBSVMModelDataParseValid : public ::testing::Test, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        const std::string template_filename = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_{}_TEMPLATE.libsvm.model", util::get_num_classes<fixture_label_type>(), fixture_classification);
        util::instantiate_template_file<fixture_label_type>(template_filename, this->filename);
    }

    /**
     * @brief Return the correct data points.
     * @return the correct data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::soa_matrix<plssvm::real_type> &get_correct_data() const noexcept { return correct_data_; }
    /**
     * @brief Return all correct weights.
     * @return the correct weights (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::vector<plssvm::real_type>> &get_correct_weights() noexcept { return correct_weights_; }
    /**
     * @brief Return the specific weight at position [@p i][@p j].
     * @details Throws if @p i or @p j are out-of-bounce.
     * @param[in] i the column index (in the model file)
     * @param[in] j the row index (in the model file)
     * @return the specific weight (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::real_type get_correct_weights(const std::size_t i, const std::size_t j) const noexcept { return correct_weights_.at(i).at(j); }

  private:
    /// The correct data points.
    plssvm::soa_matrix<plssvm::real_type> correct_data_{ { { plssvm::real_type{ -1.1178275006 }, plssvm::real_type{ -2.9087188881 }, plssvm::real_type{ 0.66638344270 }, plssvm::real_type{ 1.0978832704 } },
                                                           { plssvm::real_type{ -0.52821182989 }, plssvm::real_type{ -0.33588098497 }, plssvm::real_type{ 0.51687296030 }, plssvm::real_type{ 0.54604461446 } },
                                                           { plssvm::real_type{ 0.57650218263 }, plssvm::real_type{ 1.0140559662 }, plssvm::real_type{ 0.13009428080 }, plssvm::real_type{ 0.72619138869 } },
                                                           { plssvm::real_type{ 1.8849404372 }, plssvm::real_type{ 1.0051856432 }, plssvm::real_type{ 0.29849993305 }, plssvm::real_type{ 1.6464627049 } },
                                                           { plssvm::real_type{ -0.20981208921 }, plssvm::real_type{ 0.60276937379 }, plssvm::real_type{ -0.13086851759 }, plssvm::real_type{ 0.10805254527 } },
                                                           { plssvm::real_type{ -1.1256816276 }, plssvm::real_type{ 2.1254153434 }, plssvm::real_type{ -0.16512657655 }, plssvm::real_type{ 2.5164553141 } } },
                                                         plssvm::PADDING_SIZE,
                                                         plssvm::PADDING_SIZE };
    /// The correct weights. Might be more than are actually used in a specific test case.
    std::vector<std::vector<plssvm::real_type>> correct_weights_{
        { plssvm::real_type{ -1.8568721894e-01 }, plssvm::real_type{ 9.0116552290e-01 }, plssvm::real_type{ -2.2483112395e-01 }, plssvm::real_type{ 1.4909749921e-02 }, plssvm::real_type{ -4.5666857706e-01 }, plssvm::real_type{ -4.8888352876e-02 } },
        { plssvm::real_type{ 1.1365048527e-01 }, plssvm::real_type{ -3.2357185930e-01 }, plssvm::real_type{ 8.9871548758e-01 }, plssvm::real_type{ -7.5259922896e-02 }, plssvm::real_type{ -4.7955922738e-01 }, plssvm::real_type{ -1.3397496327e-01 } },
        { plssvm::real_type{ 2.8929914669e-02 }, plssvm::real_type{ -4.8559849173e-01 }, plssvm::real_type{ -5.6740083618e-01 }, plssvm::real_type{ 8.7841608802e-02 }, plssvm::real_type{ 9.7960957282e-01 }, plssvm::real_type{ -4.3381768383e-02 } },
        { plssvm::real_type{ 4.3106819001e-02 }, plssvm::real_type{ -9.1995171877e-02 }, plssvm::real_type{ -1.0648352745e-01 }, plssvm::real_type{ -2.7491435827e-02 }, plssvm::real_type{ -4.3381768383e-02 }, plssvm::real_type{ 2.2624508453e-01 } }
    };
};
TYPED_TEST_SUITE(LIBSVMModelDataParseValid, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelDataParseValid, read) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type expected_classification = TestFixture::fixture_classification;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    // skip the first 8 lines, i.e., the model file header using the linear kernel function
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::vector<std::size_t> num_sv_per_class = util::get_correct_model_file_num_sv_per_class<label_type>();
    const auto [num_data_points, num_features, data, alpha, classification] = plssvm::detail::io::parse_libsvm_model_data(reader, num_sv_per_class, 8);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 6);
    ASSERT_EQ(num_features, 4);
    EXPECT_EQ(classification, expected_classification);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());

    // check for correct weights
    if constexpr (expected_classification == plssvm::classification_type::oaa) {
        // OAA
        ASSERT_EQ(alpha.size(), 1);
        ASSERT_EQ(alpha.front().shape(), (std::array<std::size_t, 2>{ num_classes_for_label_type, 6 }));
        ASSERT_EQ(alpha.front().padding(), (std::array<std::size_t, 2>{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));

        switch (num_classes_for_label_type) {
            case 2:
                // ignore last two weight vectors
                this->get_correct_weights().pop_back();
                this->get_correct_weights().pop_back();
                break;
            case 3:
                // ignore last weight vector
                this->get_correct_weights().pop_back();
                break;
            case 4:
                // use full weight vector
                break;
            default:
                FAIL() << "Unreachable!";
                break;
        }

        EXPECT_EQ(alpha.front(), (plssvm::aos_matrix<plssvm::real_type>{ this->get_correct_weights(), plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    } else if constexpr (expected_classification == plssvm::classification_type::oao) {
        // OAO
        ASSERT_EQ(alpha.size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);

        std::vector<plssvm::aos_matrix<plssvm::real_type>> weights(num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
        switch (num_classes_for_label_type) {
            case 2:
                // 0vs1
                weights.front() = plssvm::aos_matrix<plssvm::real_type>{ { this->get_correct_weights().front() } };
                break;
            case 3:
                // 0vs1
                weights[0] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(0, 0), this->get_correct_weights(0, 1), this->get_correct_weights(0, 2), this->get_correct_weights(0, 3) } } };
                // 0vs2
                weights[1] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(1, 0), this->get_correct_weights(1, 1), this->get_correct_weights(0, 4), this->get_correct_weights(0, 5) } } };
                // 1vs2
                weights[2] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(1, 2), this->get_correct_weights(1, 3), this->get_correct_weights(1, 4), this->get_correct_weights(1, 5) } } };
                break;
            case 4:
                // 0vs1
                weights[0] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(0, 0), this->get_correct_weights(0, 1), this->get_correct_weights(0, 2), this->get_correct_weights(0, 3) } } };
                // 0vs2
                weights[1] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(1, 0), this->get_correct_weights(1, 1), this->get_correct_weights(0, 4) } } };
                // 0vs3
                weights[2] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(2, 0), this->get_correct_weights(2, 1), this->get_correct_weights(0, 5) } } };
                // 1vs2
                weights[3] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(1, 2), this->get_correct_weights(1, 3), this->get_correct_weights(1, 4) } } };
                // 1vs3
                weights[4] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(2, 2), this->get_correct_weights(2, 3), this->get_correct_weights(1, 5) } } };
                // 2vs3
                weights[5] = plssvm::aos_matrix<plssvm::real_type>{ { { this->get_correct_weights(2, 4), this->get_correct_weights(2, 5) } } };
                break;
            default:
                FAIL() << "Unreachable!";
                break;
        }
        // add padding to each matrix (theoretically expensive, but matrices are tiny)
        for (plssvm::aos_matrix<plssvm::real_type> &matr : weights) {
            matr = plssvm::aos_matrix<plssvm::real_type>{ matr, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE, };
        }

        EXPECT_EQ(alpha, weights);
    } else {
        FAIL() << "unknown classification type";
    }
}