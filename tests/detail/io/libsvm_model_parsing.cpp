/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to parsing the support vectors from a LIBSVM model file.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "../../custom_test_macros.hpp"      // EXPECT_FLOATING_POINT_EQ, EXPECT_THROW_WHAT
#include "../../naming.hpp"                  // naming::real_type_label_type_combination_to_name
#include "../../types_to_test.hpp"           // util::real_type_label_type_combination_gtest
#include "../../utility.hpp"                 // util::{temporary_file, instantiate_template_model_file, get_correct_model_file_labels, get_distinct_label, generate_specific_matrix}

#include "fmt/core.h"                        // fmt::format
#include "gmock/gmock-matchers.h"            // ::testing::HasSubstr
#include "gtest/gtest.h"                     // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ASSERT_EQ, GTEST_FAIL
                                             // ::testing::{Test, Types, Values}

#include <cstddef>                           // std::size_t
#include <string>                            // std::string
#include <tuple>                             // std::ignore
#include <vector>                            // std::vector

template <typename T>
class LIBSVMModelHeaderParse : public ::testing::Test {};

template <typename T>
class LIBSVMModelHeaderParseValid : public LIBSVMModelHeaderParse<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderParseValid, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(LIBSVMModelHeaderParseValid, read_linear) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using size_type = std::size_t;

    // create temporary file
    const util::temporary_file template_file{};
    util::instantiate_template_model_file<label_type>(PLSSVM_TEST_PATH "/data/model/6x4_linear_TEMPLATE.libsvm.model", template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::linear);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_TRUE(params.gamma.is_default());
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_FLOATING_POINT_EQ(rho, real_type{ 0.37330625882191915 });
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(header_lines, 8);
}
TYPED_TEST(LIBSVMModelHeaderParseValid, read_polynomial) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using size_type = std::size_t;

    // create temporary file
    const util::temporary_file template_file{};
    util::instantiate_template_model_file<label_type>(PLSSVM_TEST_PATH "/data/model/6x4_polynomial_TEMPLATE.libsvm.model", template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_FALSE(params.degree.is_default());
    EXPECT_EQ(params.degree.value(), 2);
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(params.gamma.value(), 0.25);
    EXPECT_FALSE(params.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(params.coef0.value(), 1.5);
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_FLOATING_POINT_EQ(rho, real_type{ 0.37330625882191915 });
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(header_lines, 11);
}
TYPED_TEST(LIBSVMModelHeaderParseValid, read_rbf) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using size_type = std::size_t;

    // create temporary file
    const util::temporary_file template_file{};
    util::instantiate_template_model_file<label_type>(PLSSVM_TEST_PATH "/data/model/6x4_rbf_TEMPLATE.libsvm.model", template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::rbf);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(params.gamma.value(), 0.025);
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_FLOATING_POINT_EQ(rho, real_type{ 0.37330625882191915 });
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(header_lines, 9);
}

template <typename T>
class LIBSVMModelHeaderParseInvalid : public LIBSVMModelHeaderParse<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderParseInvalid, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_svm_type) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can only use c_svc as svm_type, but 'nu_svc' was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_kernel_type) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized kernel type 'foo'!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_total_sv) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of support vectors must be greater than 0!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two labels must be set, but only 1 label ([1]) was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_nr_sv) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two nr_sv must be set, but only 1 ([5]) was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, unrecognized_header_entry) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/unrecognized_header_entry.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry 'invalid entry'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_svm_type) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_kernel_type) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing kernel_type!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_linear_kernel) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_gamma_in_linear_kernel) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_gamma_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_linear_kernel) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_rbf_kernel) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the radial basis function kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_rbf_kernel) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the radial basis function kernel!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_class) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of different classes nr_class!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_total_sv) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing total number of support vectors total_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_rho) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing rho value!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing class label specification!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_label_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_label_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of classes (nr_class) is 2, but the provided number of different labels is 3 (label)!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_sv) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of support vectors per class nr_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_nr_sv_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of classes (nr_class) is 2, but the provided number of different labels is 3 (nr_sv)!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, total_sv_and_nr_sv_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/total_sv_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The total number of support vectors is 5, but the sum of nr_sv is 6!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_sv) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry '-0.17609610490769723 1:-1.117828e+00 2:-2.908719e+00 3:6.663834e-01 4:1.097883e+00'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_support_vectors.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no support vectors are given or SV is missing!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, same_class_multiple_times) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/same_class_multiple_times.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Provided 2 labels but only 1 of them was/where unique!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, empty) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}

template <typename T>
class LIBSVMModelWriteBase : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {};

template <typename T>
class LIBSVMModelHeaderWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWrite, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(LIBSVMModelHeaderWrite, write_linear) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::vector<real_type>> data = util::generate_specific_matrix<real_type>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{};
    const real_type rho = 3.14159265359;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(3), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.size()));
    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(6), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(7), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_polynomial) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::vector<real_type>> data = util::generate_specific_matrix<real_type>(label.size(), 3);

    // create necessary parameter
    plssvm::parameter params{};
    params.kernel_type = plssvm::kernel_function_type::polynomial;
    params.degree = 3;
    params.gamma = 2.2;
    params.coef0 = 4.4;
    const real_type rho = 2.71828;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 11);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type polynomial");
    EXPECT_EQ(reader.line(2), "degree 3");
    EXPECT_EQ(reader.line(3), "gamma 2.2");
    EXPECT_EQ(reader.line(4), "coef0 4.4");
    EXPECT_EQ(reader.line(5), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(6), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(7), fmt::format("total_sv {}", data.size()));
    EXPECT_EQ(reader.line(8), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(9), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(10), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_rbf) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::vector<real_type>> data = util::generate_specific_matrix<real_type>(label.size(), 3);

    // create necessary parameter
    plssvm::parameter params{};
    params.kernel_type = plssvm::kernel_function_type::rbf;
    params.gamma = 0.4;
    const real_type rho = 1.41421;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 9);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type rbf");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(4), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(5), fmt::format("total_sv {}", data.size()));
    EXPECT_EQ(reader.line(6), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(7), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(8), "SV");
}

template <typename T>
class LIBSVMModelDataWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelDataWrite, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(LIBSVMModelDataWrite, write) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::vector<real_type>> data = util::generate_specific_matrix<real_type>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{};
    const real_type rho = 3.1415;
    std::vector<real_type> alpha(data.size());
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        alpha[i] = -static_cast<real_type>(i + 1);
    }
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    std::cerr << reader.buffer() << std::endl;
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8 + data.size());  // header lines + data
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(3), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.size()));
    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(6), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(7), "SV");

    // sort data points into buckets for each label type
    std::unordered_map<label_type, std::vector<std::pair<real_type, std::vector<real_type>>>> data_per_label;
    for (std::size_t i = 0; i < label.size(); ++i) {
        if (!plssvm::detail::contains(data_per_label, label[i])) {
            data_per_label[label[i]] = std::vector<std::pair<real_type, std::vector<real_type>>>{};
        }
        data_per_label[label[i]].emplace_back(std::make_pair(alpha[i], data[i]));
    }

    // check lines for correctness
    std::size_t idx = 8;
    for (std::size_t i = 0; i < distinct_label.size(); ++i) {
        for (std::size_t j = 0; j < num_sv[i]; ++j) {
            // check if any of the data would match the current line
            bool line_found = false;
            for (const auto &[alpha_val, vec] : data_per_label[distinct_label[i]]) {
                const std::string line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", alpha_val, vec[0], vec[1], vec[2]);
                if (reader.line(idx) == line) {
                    line_found = true;
                }
            }
            if (!line_found) {
                GTEST_FAIL() << fmt::format("Couldn't find line the line '{}' from the output file in the provided data set.", reader.line(idx));
            }
            ++idx;
        }

    }
}

template <typename T>
class LIBSVMModelWriteDeathTest : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelWriteDeathTest, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(LIBSVMModelWriteDeathTest, write_header_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create necessary parameter
    const plssvm::parameter params{};
    const real_type rho{};
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ { real_type{ 0.0 } } } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_without_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create necessary parameter
    const plssvm::parameter params{};
    const real_type rho{};
    const std::vector<real_type> alpha{ real_type{ 0.1 } };
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ { real_type{ 0.0 } } } };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelWriteDeathTest, num_alphas_and_num_data_points_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create necessary parameter
    const plssvm::parameter params{};
    const real_type rho{};
    const std::vector<real_type> alpha{ real_type{ 0.1 } };
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>().first;
    const std::vector<std::vector<real_type>> data = util::generate_specific_matrix<real_type>(label.size(), 1);

    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::move(label) };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 ::testing::HasSubstr(fmt::format("The number of weights (1) doesn't match the number of data points ({})!", data.size())));
}