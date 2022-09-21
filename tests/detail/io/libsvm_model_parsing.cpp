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

#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "../../utility.hpp"  // util::create_temp_file, EXPECT_THROW_WHAT

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_P, TYPED_TEST, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, GTEST_FAIL
                                   // ::testing::{Test, Types, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <string>       // std::string
#include <tuple>        // std::ignore
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

// struct for the used type combinations
template <typename T, typename U>
struct type_combinations {
    using real_type = T;
    using label_type = U;
};

// the floating point and label types combinations to test
using type_combinations_types = ::testing::Types<
    type_combinations<float, int>,
    type_combinations<float, std::string>,
    type_combinations<double, int>,
    type_combinations<double, std::string>>;

TEST(LIBSVMModelHeaderParseValid, read_linear) {
    using real_type = double;
    using label_type = int;
    using size_type = std::size_t;

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel.is_default());
    EXPECT_EQ(params.kernel.value(), plssvm::kernel_type::linear);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_TRUE(params.gamma.is_default());
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_EQ(rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
    EXPECT_EQ(label, (std::vector<label_type>{ 1, 1, -1, -1, -1 }));
    EXPECT_EQ(header_lines, 8);
}
TEST(LIBSVMModelHeaderParseValid, read_polynomial) {
    using real_type = double;
    using label_type = int;
    using size_type = std::size_t;

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/model/5x4_polynomial.libsvm.model" };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel.is_default());
    EXPECT_EQ(params.kernel.value(), plssvm::kernel_type::polynomial);
    EXPECT_FALSE(params.degree.is_default());
    EXPECT_EQ(params.degree.value(), 2);
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_EQ(params.gamma.value(), 0.25);
    EXPECT_FALSE(params.coef0.is_default());
    EXPECT_EQ(params.coef0.value(), 1.5);
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_EQ(rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
    EXPECT_EQ(label, (std::vector<label_type>{ 1, 1, -1, -1, -1 }));
    EXPECT_EQ(header_lines, 11);
}
TEST(LIBSVMModelHeaderParseValid, read_rbf) {
    using real_type = double;
    using label_type = int;
    using size_type = std::size_t;

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/model/5x4_rbf.libsvm.model" };
    reader.read_lines('#');
    const auto &[params, rho, label, header_lines] = plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines());

    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel.is_default());
    EXPECT_EQ(params.kernel.value(), plssvm::kernel_type::rbf);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_EQ(params.gamma.value(), 0.025);
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    EXPECT_EQ(rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
    EXPECT_EQ(label, (std::vector<label_type>{ 1, 1, -1, -1, -1 }));
    EXPECT_EQ(header_lines, 9);
}

template <typename T>
class LIBSVMModelHeaderParse : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMModelHeaderParse, type_combinations_types);

TYPED_TEST(LIBSVMModelHeaderParse, wrong_svm_type) {
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
TYPED_TEST(LIBSVMModelHeaderParse, wrong_kernel_type) {
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
TYPED_TEST(LIBSVMModelHeaderParse, wrong_total_sv) {
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
TYPED_TEST(LIBSVMModelHeaderParse, too_few_label) {
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
TYPED_TEST(LIBSVMModelHeaderParse, too_few_nr_sv) {
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
TYPED_TEST(LIBSVMModelHeaderParse, unrecognized_header_entry) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_svm_type) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_kernel_type) {
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

TYPED_TEST(LIBSVMModelHeaderParse, explicit_degree_in_linear_kernel) {
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
TYPED_TEST(LIBSVMModelHeaderParse, explicit_gamma_in_linear_kernel) {
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
TYPED_TEST(LIBSVMModelHeaderParse, explicit_coef0_in_linear_kernel) {
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
TYPED_TEST(LIBSVMModelHeaderParse, explicit_degree_in_rbf_kernel) {
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
TYPED_TEST(LIBSVMModelHeaderParse, explicit_coef0_in_rbf_kernel) {
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

TYPED_TEST(LIBSVMModelHeaderParse, missing_nr_class) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_total_sv) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_rho) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_label) {
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
TYPED_TEST(LIBSVMModelHeaderParse, nr_class_and_label_mismatch) {
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_nr_sv) {
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
TYPED_TEST(LIBSVMModelHeaderParse, nr_class_and_nr_sv_mismatch) {
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
TYPED_TEST(LIBSVMModelHeaderParse, total_sv_and_nr_sv_mismatch) {
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
TYPED_TEST(LIBSVMModelHeaderParse, too_many_classes) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::real_type;
    using size_type = std::size_t;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_many_classes.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<real_type, label_type, size_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Currently only binary classification is supported, but 3 different label where given!");
}
TYPED_TEST(LIBSVMModelHeaderParse, missing_sv) {
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
TYPED_TEST(LIBSVMModelHeaderParse, empty) {
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