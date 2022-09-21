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

#include "../../utility.hpp"  // util::create_temp_file, EXPECT_THROW_WHAT

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ASSERT_EQ, GTEST_FAIL
                          // ::testing::{Test, Types, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string
#include <tuple>        // std::ignore
#include <type_traits>  // std::is_same_v
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
TYPED_TEST(LIBSVMModelHeaderParse, missing_support_vectors) {
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

template <typename T>
class LIBSVMModelWriteBase : public ::testing::Test {
  protected:
    void SetUp() override {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());

        // create a temporary file containing the scaling factors
        filename = util::create_temp_file();
    }
    void TearDown() override {
        // remove the temporary file at the end
        std::filesystem::remove(filename);

        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

    std::string filename;

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};

template <typename T>
class LIBSVMModelHeaderWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWrite, type_combinations_types);

TYPED_TEST(LIBSVMModelHeaderWrite, write_linear) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ -1, 1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "cat", "dog", "cat" };
    }

    // create necessary parameter
    const plssvm::parameter<real_type> params{};
    const real_type rho = 3.14159265359;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(label_order, (std::vector<int>{ -1, 1 }));
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(label_order, (std::vector<std::string>{ "cat", "dog" }));
    }

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), "nr_class 2");
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(reader.line(3), "label -1 1");
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(reader.line(3), "label cat dog");
    }
    EXPECT_EQ(reader.line(4), "total_sv 3");
    EXPECT_EQ(reader.line(5), "nr_sv 2 1");
    EXPECT_EQ(reader.line(6), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(7), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_polynomial) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ 1, 2, 2 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "Cat", "Dog", "Dog" };
    }

    // create necessary parameter
    plssvm::parameter<real_type> params{};
    params.kernel = plssvm::kernel_type::polynomial;
    params.degree = 3;
    params.gamma = real_type{ 2.2 };
    params.coef0 = real_type{ 4.4 };
    const real_type rho = 2.71828;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(label_order, (std::vector<int>{ 1, 2 }));
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(label_order, (std::vector<std::string>{ "Cat", "Dog" }));
    }

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
    EXPECT_EQ(reader.line(5), "nr_class 2");
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(reader.line(6), "label 1 2");
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(reader.line(6), "label Cat Dog");
    }
    EXPECT_EQ(reader.line(7), "total_sv 3");
    EXPECT_EQ(reader.line(8), "nr_sv 1 2");
    EXPECT_EQ(reader.line(9), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(10), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_rbf) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ 1, -1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "B", "A", "A" };
    }

    // create necessary parameter
    plssvm::parameter<real_type> params{};
    params.kernel = plssvm::kernel_type::rbf;
    params.gamma = real_type{ 0.4 };
    const real_type rho = 1.41421;
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    const std::vector<label_type>& label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(label_order, (std::vector<int>{ -1, 1 }));
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(label_order, (std::vector<std::string>{ "A", "B" }));
    }

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 9);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type rbf");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), "nr_class 2");
    if constexpr (std::is_same_v<label_type, int>) {
        EXPECT_EQ(reader.line(4), "label -1 1");
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        EXPECT_EQ(reader.line(4), "label A B");
    }
    EXPECT_EQ(reader.line(5), "total_sv 3");
    EXPECT_EQ(reader.line(6), "nr_sv 2 1");
    EXPECT_EQ(reader.line(7), fmt::format("rho {}", rho));
    EXPECT_EQ(reader.line(8), "SV");
}

template <typename T>
class LIBSVMModelDataWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelDataWrite, type_combinations_types);

TYPED_TEST(LIBSVMModelDataWrite, write) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ -1, 1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "cat", "dog", "cat" };
    }

    // create necessary parameter
    const std::vector<real_type> alpha{ -0.1, -0.2, -0.3 };
    const plssvm::data_set<real_type, label_type> data_set{ std::vector<std::vector<real_type>>{ data }, std::vector<label_type>{ label } };
    std::vector<label_type> label_order{};
    if constexpr (std::is_same_v<label_type, int>) {
        label_order = std::vector<int>{ -1, 1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label_order = std::vector<std::string>{ "cat", "dog" };
    }

    // write the LIBSVM model file
    plssvm::detail::io::write_libsvm_model_data(out, alpha, data_set, label_order);
    out.close();

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), data.size());
    // at first the two lines with the first label must have been written, the internal order may be random
    for (const std::size_t i : { 0, 2 }) {
        const std::string line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", alpha[i], data[i][0], data[i][1], data[i][2]);
        bool line_found = false;
        for (std::size_t j = 0; j < 2; ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            GTEST_FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }

    // then the lines with the second label
    ASSERT_EQ(reader.line(2), fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", alpha[1], data[1][0], data[1][1], data[1][2]));
}