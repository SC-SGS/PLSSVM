/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing a LIBSVM regression model file with an invalid header.
 */

#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/exceptions/exceptions.hpp"                      // plssvm::invalid_file_format_exception

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER

#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST

#include <string>  // std::string
#include <tuple>   // std::ignore

TEST(LIBSVMRegressionModelHeaderParseInvalid, WrongSvmType) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/wrong_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can only use c_svr as svm_type, but 'epsilon_svr' was given!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, WrongKernelType) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/wrong_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized kernel type 'foo'!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, WrongTotalSv) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/wrong_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of support vectors must be greater than 0!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, EmptyRho) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/empty_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least one rho value must be set, but none was given!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, UnrecognizedHeaderEntry) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/unrecognized_header_entry.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry 'invalid entry'! Maybe SV is missing?");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingSvmType) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingKernelType) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing kernel_type!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitDegreeInLinearKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_degree_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the linear kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitGammaInLinearKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_gamma_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitCoef0InLinearKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_coef0_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitDegreeInRadialBasisFunctionKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_degree_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the rbf kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitCoef0InRadialBasisFunctionKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_coef0_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the rbf kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitDegreeInSigmoidKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_degree_in_sigmoid_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the sigmoid kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitDegreeInLaplacianKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_degree_in_laplacian_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the laplacian kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitCoef0InLaplacianKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_coef0_in_laplacian_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the laplacian kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitDegreeInChiSquaredKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_degree_in_chi_squared_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the chi_squared kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, ExplicitCoef0InChiSquaredKernel) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/explicit_coef0_in_chi_squared_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the chi_squared kernel!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingNrClass) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_nr_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of different classes nr_class!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingTotalSv) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing total number of support vectors total_sv!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingRho) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing rho values!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingSv) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry '-1.8568721894e-01 1:-1.1178275006e+00 2:-2.9087188881e+00 3:6.6638344270e-01 4:1.0978832704e+00'! Maybe SV is missing?");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, MissingSupportVectors) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_support_vectors.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no support vectors are given or SV is missing!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, Empty) {
    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, TooFewSvAccordingToHeader) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/too_few_sv_according_to_header.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Found 5 support vectors, but it should be 6!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, TooManySvAccordingToHeader) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/too_many_sv_according_to_header.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Found 7 support vectors, but it should be 6!");
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, WrongNrClass) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/wrong_nr_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT_MATCHER(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                              plssvm::invalid_file_format_exception,
                              ::testing::HasSubstr("The number of classes (nr_class) is 3, but must be 2!"));
}

TEST(LIBSVMRegressionModelHeaderParseInvalid, WrongNumRho) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/wrong_num_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Provided 2 rho values but only one is needed!");
}
