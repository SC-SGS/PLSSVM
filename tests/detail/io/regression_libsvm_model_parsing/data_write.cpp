/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for writing LIBSVM regression model file data section.
 */

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"               // plssvm::regression_data_set
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/kernel_function_types.hpp"                      // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                     // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                                  // plssvm::parameter
#include "plssvm/shape.hpp"                                      // plssvm::shape

#include "tests/naming.hpp"         // naming::parameter_definition_to_name
#include "tests/types_to_test.hpp"  // util::regression_label_type_gtest
#include "tests/utility.hpp"        // util::{get_distinct_label, get_correct_model_file_labels, get_correct_model_file_num_sv_per_class,
                                    // generate_random_matrix, get_num_classes, generate_random_vector}

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_DEATH, ASSERT_EQ, FAIL, SUCCEED, ::testing::Test

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

template <typename T>
class LIBSVMRegressionModelDataWrite : public ::testing::Test,
                                       private util::redirect_output<>,
                                       protected util::temporary_file {
  public:
    /**
     * @brief Return the used MPI communicator.
     * @return the MPI communicator (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::mpi::communicator get_comm() const noexcept { return comm_; }

  private:
    /// The MPI communicator (unused during testing since we do not support MPI runtime tests).
    plssvm::mpi::communicator comm_{};
};

TYPED_TEST_SUITE(LIBSVMRegressionModelDataWrite, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelDataWrite, write) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.1415 } };
    std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 1, data.num_rows() }) };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model file
    plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), params, rho, alpha, data_set);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    const std::size_t header_offset{ 6 };

    // check the written data
    ASSERT_EQ(reader.num_lines(), header_offset + data.num_rows());  // header lines + data
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), "nr_class 2");
    EXPECT_EQ(reader.line(3), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(4), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(5), "SV");

    // iterate over all support vectors
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        // check whether the necessary row is present in the data set
        const std::string correct_line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", alpha.front()(0, i), data(i, 0), data(i, 1), data(i, 2));
        int line_found{ 0 };

        std::string_view read_line{};
        for (std::size_t k = 0; k < reader.num_lines() - header_offset; ++k) {
            read_line = reader.line(header_offset + k);

            if (read_line == correct_line) {
                ++line_found;
            }
        }

        // check, how often the line in the file was found in the original data
        if (line_found == 0) {
            FAIL() << fmt::format("Couldn't find the line '{}' ({}) from the output file in the provided data set.", read_line, i);
        } else if (line_found > 1) {
            FAIL() << fmt::format("Could find the line '{}' ({}) from the output file in the provided data set multiple times.", read_line, i);
        }
    }
    SUCCEED();
}

TYPED_TEST(LIBSVMRegressionModelDataWrite, write_without_label) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.1415 } };
    std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 1, data.num_rows() }) };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model file
    plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), params, rho, alpha, data_set);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    const std::size_t header_offset{ 6 };

    // check the written data
    ASSERT_EQ(reader.num_lines(), header_offset + data.num_rows());  // header lines + data
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), "nr_class 2");
    EXPECT_EQ(reader.line(3), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(4), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(5), "SV");

    // iterate over all support vectors
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        // check whether the necessary row is present in the data set
        const std::string correct_line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", alpha.front()(0, i), data(i, 0), data(i, 1), data(i, 2));
        int line_found{ 0 };

        std::string_view read_line{};
        for (std::size_t k = 0; k < reader.num_lines() - header_offset; ++k) {
            read_line = reader.line(header_offset + k);

            if (read_line == correct_line) {
                ++line_found;
            }
        }

        // check, how often the line in the file was found in the original data
        if (line_found == 0) {
            FAIL() << fmt::format("Couldn't find the line '{}' ({}) from the output file in the provided data set.", read_line, i);
        } else if (line_found > 1) {
            FAIL() << fmt::format("Could find the line '{}' ({}) from the output file in the provided data set multiple times.", read_line, i);
        }
    }
    SUCCEED();
}

template <typename T>
class LIBSVMRegressionModelDataWriteDeathTest : public LIBSVMRegressionModelDataWrite<T> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    /**
     * @brief Return the default parameter.
     * @return the parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::parameter &get_params() const noexcept { return params_; }

    /**
     * @brief Return the rho values.
     * @details The size depends on the used classification type and number of classes.
     * @return the rho values (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::real_type> &get_rho() const noexcept { return rho_; }

    /**
     * @brief Return the weights.
     * @details The shape of the vector and the containing matrices depend on the used classification type and number of classes.
     * @return the weights (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::aos_matrix<plssvm::real_type>> &get_alpha() const noexcept { return alpha_; }

    /**
     * @brief Return the data set containing all support vectors.
     * @return the support vectors (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::regression_data_set<fixture_label_type> &get_data_set() const noexcept { return data_set_; }

  private:
    /// The default parameters.
    plssvm::parameter params_{};
    /// The rho vector; size depending on used classification type and number of classes.
    std::vector<plssvm::real_type> rho_{ plssvm::real_type{ 3.1415 } };
    /// The weights; shape of the vector and the containing matrices depending on used classification type and number of classes.
    std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha_{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 1, 6 }) };
    /// The support vectors.
    plssvm::regression_data_set<fixture_label_type> data_set_{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 6, 2 }), util::generate_random_vector<fixture_label_type>(6) };
};

TYPED_TEST_SUITE(LIBSVMRegressionModelDataWriteDeathTest, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelDataWriteDeathTest, empty_filename) {
    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data_regression("", this->get_comm(), this->get_params(), this->get_rho(), this->get_alpha(), this->get_data_set())),
                 "The provided model filename must not be empty!");
}

TYPED_TEST(LIBSVMRegressionModelDataWriteDeathTest, invalid_number_of_rho_values) {
    // create invalid parameter
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(42);

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), this->get_params(), rho, this->get_alpha(), this->get_data_set())),
                 "The number of rho values is 42 but must be exactly 1!");
}

TYPED_TEST(LIBSVMRegressionModelDataWriteDeathTest, invalid_alpha_vector) {
    {
        // alpha vector too large
        const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha(2);
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), this->get_params(), this->get_rho(), alpha, this->get_data_set())),
                     "The alpha vector may only contain one matrix as entry, but has 2!");
    }
    {
        // invalid number of rows in matrix
        const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ plssvm::aos_matrix<plssvm::real_type>{ plssvm::shape{ 42, 6 } } };
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), this->get_params(), this->get_rho(), alpha, this->get_data_set())),
                     "The number of rows in the matrix must be 1, but is 42!");
    }
    {
        // invalid number of columns in matrix
        const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ plssvm::aos_matrix<plssvm::real_type>{ plssvm::shape{ 1, 42 } } };
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data_regression(this->filename, this->get_comm(), this->get_params(), this->get_rho(), alpha, this->get_data_set())),
                     ::testing::HasSubstr("The number of weights (42) must be equal to the number of support vectors (6)!"));
    }
}
