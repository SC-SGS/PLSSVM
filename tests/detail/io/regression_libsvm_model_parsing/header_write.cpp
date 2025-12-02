/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for writing a LIBSVM regression model file header.
 */

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"               // plssvm::regression_data_set
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/kernel_function_types.hpp"                      // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                                     // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                                  // plssvm::parameter
#include "plssvm/shape.hpp"                                      // plssvm::shape

#include "tests/naming.hpp"         // naming::label_type_to_name
#include "tests/types_to_test.hpp"  // util::regression_label_type_gtest
#include "tests/utility.hpp"        // util::{temporary_file, generate_specific_matrix}

#include "fmt/format.h"   // fmt::format
#include "fmt/os.h"       // fmt::ostream, fmt::output_file
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_DEATH, ASSERT_EQ, ::testing::Test

#include <vector>  // std::vector

template <typename T>
class LIBSVMRegressionModelHeaderWrite : public ::testing::Test,
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

TYPED_TEST_SUITE(LIBSVMRegressionModelHeaderWrite, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteLinear) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 6);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), "nr_class 2");
    EXPECT_EQ(reader.line(3), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(4), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(5), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteLinearWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 6);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), "nr_class 2");
    EXPECT_EQ(reader.line(3), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(4), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(5), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WritePolynomial) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::degree = 3, plssvm::gamma = 2.2, plssvm::coef0 = 4.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 9);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type polynomial");
    EXPECT_EQ(reader.line(2), "degree 3");
    EXPECT_EQ(reader.line(3), "gamma 2.2");
    EXPECT_EQ(reader.line(4), "coef0 4.4");
    EXPECT_EQ(reader.line(5), "nr_class 2");
    EXPECT_EQ(reader.line(6), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(7), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(8), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WritePolynomialWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::degree = 3, plssvm::gamma = 2.2, plssvm::coef0 = 4.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 9);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type polynomial");
    EXPECT_EQ(reader.line(2), "degree 3");
    EXPECT_EQ(reader.line(3), "gamma 2.2");
    EXPECT_EQ(reader.line(4), "coef0 4.4");
    EXPECT_EQ(reader.line(5), "nr_class 2");
    EXPECT_EQ(reader.line(6), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(7), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(8), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteRadialBasisFunction) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type rbf");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteRadialBasisFunctionWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type rbf");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteSigmoid) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::sigmoid, plssvm::gamma = 0.4, plssvm::coef0 = 4.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type sigmoid");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "coef0 4.4");
    EXPECT_EQ(reader.line(4), "nr_class 2");
    EXPECT_EQ(reader.line(5), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(6), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(7), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteSigmoidWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::sigmoid, plssvm::gamma = 0.4, plssvm::coef0 = 4.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type sigmoid");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "coef0 4.4");
    EXPECT_EQ(reader.line(4), "nr_class 2");
    EXPECT_EQ(reader.line(5), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(6), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(7), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteLaplacian) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::laplacian, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type laplacian");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteLaplacianWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::laplacian, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type laplacian");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteChiSquared) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> label = util::generate_random_vector<label_type>(6);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ label.size(), 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::chi_squared, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data, label };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type chi_squared");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

TYPED_TEST(LIBSVMRegressionModelHeaderWrite, WriteChiSquaredWithoutLabel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 3 });

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::chi_squared, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 3.14159265359 } };
    const plssvm::regression_data_set<label_type> data_set{ data };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 7);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svr");
    EXPECT_EQ(reader.line(1), "kernel_type chi_squared");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("rho {:.10e}", rho.front()));
    EXPECT_EQ(reader.line(6), "SV");
}

template <typename T>
class LIBSVMRegressionModelHeaderWriteDeathTest : public LIBSVMRegressionModelHeaderWrite<T>,
                                                  private util::redirect_output<> { };

TYPED_TEST_SUITE(LIBSVMRegressionModelHeaderWriteDeathTest, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelHeaderWriteDeathTest, WriteHeaderInvalidNumberOfRhoValues) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> rho{};
    const plssvm::regression_data_set<label_type> data_set{ std::vector<std::vector<plssvm::real_type>>{ { plssvm::real_type{ 0.0 } } },
                                                            std::vector<label_type>{ 0 } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_header_regression(out, this->get_comm(), params, rho, data_set)),
                 ::testing::HasSubstr("Exactly one rho value must be provided!"));
}
