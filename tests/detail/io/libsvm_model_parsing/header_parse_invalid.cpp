/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing a LIBSVM model file with an invalid header.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::label_type_to_name
#include "types_to_test.hpp"       // util::label_type_gtest
#include "utility.hpp"             // util::get_num_classes

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, SUCCEED
                          // ::testing::Test

#include <string>  // std::string
#include <tuple>   // std::ignore

template <typename T>
class LIBSVMModelHeaderParseInvalid : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMModelHeaderParseInvalid, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_svm_type) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can only use c_svc as svm_type, but 'nu_svc' was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_kernel_type) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized kernel type 'foo'!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_total_sv) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of support vectors must be greater than 0!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_label) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two labels must be set, but only one label was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_nr_sv) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two nr_sv must be set, but only one was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, empty_rho) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/empty_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least one rho value must be set, but none was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_rho) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    if (util::get_num_classes<label_type>() == 2) {
        SUCCEED() << "Test not applicable for the label types with only two distinct classes.";
    } else {
        // parse the LIBSVM model file
        const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_rho.libsvm.model";
        plssvm::detail::io::file_reader reader{ filename };
        reader.read_lines('#');
        EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                          plssvm::invalid_file_format_exception,
                          "Provided 1 rho values but at least min(3, 3) = 3 are needed!");
    }
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, unrecognized_header_entry) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/unrecognized_header_entry.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry 'invalid entry'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_svm_type) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_kernel_type) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing kernel_type!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_linear_kernel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_gamma_in_linear_kernel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_gamma_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_linear_kernel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_rbf_kernel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the radial basis function kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_rbf_kernel) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the radial basis function kernel!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_class) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of different classes nr_class!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_total_sv) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing total number of support vectors total_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_rho) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing rho values!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_label) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing class label specification!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_label_mismatch) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_label_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of classes (nr_class) is 3, but the provided number of different labels is 2 (label)!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_sv) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of support vectors per class nr_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_nr_sv_mismatch) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of classes (nr_class) is 2, but the provided number of different labels is 3 (nr_sv)!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, total_sv_and_nr_sv_mismatch) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/total_sv_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The total number of support vectors is 5, but the sum of nr_sv is 6!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_sv) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry '-1.8568721894e-01 1:-1.1178275006e+00 2:-2.9087188881e+00 3:6.6638344270e-01 4:1.0978832704e+00'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_support_vectors) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_support_vectors.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no support vectors are given or SV is missing!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, same_class_multiple_times) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/same_class_multiple_times.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Provided 2 labels but only 1 of them was/where unique!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, empty) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}