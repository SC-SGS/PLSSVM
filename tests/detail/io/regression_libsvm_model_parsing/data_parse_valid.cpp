/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing a valid LIBSVM regression model file data section.
 */

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/matrix.hpp"                                     // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/shape.hpp"                                      // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR
#include "tests/utility.hpp"             // util::{temporary_file, instantiate_template_file}

#include "gtest/gtest.h"  // TEST_F, EXPECT_EQ, ASSERT_EQ, FAIL, ::testing::Test

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

class LIBSVMRegressionModelDataParseValid : public ::testing::Test,
                                            protected util::temporary_file {
  protected:
    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        // Note: the real_type doesn't matter here
        util::instantiate_template_file<plssvm::real_type>(PLSSVM_TEST_PATH "/data/model/regression/6x4_TEMPLATE.libsvm.model", this->filename);
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
    [[nodiscard]] const plssvm::aos_matrix<plssvm::real_type> &get_correct_weights() noexcept { return correct_weights_; }

  private:
    /// The correct data points.
    plssvm::soa_matrix<plssvm::real_type> correct_data_{ { { plssvm::real_type{ -1.1178275006 }, plssvm::real_type{ -2.9087188881 }, plssvm::real_type{ 0.66638344270 }, plssvm::real_type{ 1.0978832704 } },
                                                           { plssvm::real_type{ -0.52821182989 }, plssvm::real_type{ -0.33588098497 }, plssvm::real_type{ 0.51687296030 }, plssvm::real_type{ 0.54604461446 } },
                                                           { plssvm::real_type{ 0.57650218263 }, plssvm::real_type{ 1.0140559662 }, plssvm::real_type{ 0.13009428080 }, plssvm::real_type{ 0.72619138869 } },
                                                           { plssvm::real_type{ 1.8849404372 }, plssvm::real_type{ 1.0051856432 }, plssvm::real_type{ 0.29849993305 }, plssvm::real_type{ 1.6464627049 } },
                                                           { plssvm::real_type{ -0.20981208921 }, plssvm::real_type{ 0.60276937379 }, plssvm::real_type{ -0.13086851759 }, plssvm::real_type{ 0.10805254527 } },
                                                           { plssvm::real_type{ -1.1256816276 }, plssvm::real_type{ 2.1254153434 }, plssvm::real_type{ -0.16512657655 }, plssvm::real_type{ 2.5164553141 } } } };
    /// The correct weights.
    plssvm::aos_matrix<plssvm::real_type> correct_weights_{ { { plssvm::real_type{ -1.8568721894e-01 }, plssvm::real_type{ 9.0116552290e-01 }, plssvm::real_type{ -2.2483112395e-01 }, plssvm::real_type{ 1.4909749921e-02 }, plssvm::real_type{ -4.5666857706e-01 }, plssvm::real_type{ -4.8888352876e-02 } } } };
};

TEST_F(LIBSVMRegressionModelDataParseValid, read) {
    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    // skip the first 8 lines, i.e., the model file header using the linear kernel function
    const auto [data, alpha] = plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6);

    // check for correct sizes
    ASSERT_EQ(data.num_rows(), 6);
    ASSERT_EQ(data.num_cols(), 4);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());

    // check for correct weights
    ASSERT_EQ(alpha.size(), 1);
    ASSERT_EQ(alpha.front().shape(), (plssvm::shape{ 1, 6 }));
    EXPECT_EQ(alpha.front(), this->get_correct_weights());
}
