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
#include "../../naming.hpp"                  // naming::label_type_to_name
#include "../../types_to_test.hpp"           // util::label_type_gtest
#include "../../utility.hpp"                 // util::{temporary_file, get_correct_model_file_labels, get_distinct_label, generate_specific_matrix}

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
TYPED_TEST_SUITE(LIBSVMModelHeaderParseValid, util::label_type_gtest , naming::label_type_to_name);

TYPED_TEST(LIBSVMModelHeaderParseValid, read_linear) {
    using label_type = TypeParam;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

    // TODO: test num_sv_per_class
    // check for correctness
    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::linear);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_TRUE(params.gamma.is_default());
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());
    // check remaining values
    const std::vector<plssvm::real_type> all_rhos{ plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 } };
    ASSERT_EQ(rho.size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, all_rhos);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());
    EXPECT_EQ(header_lines, 8);
}
TYPED_TEST(LIBSVMModelHeaderParseValid, read_polynomial) {
    using label_type = TypeParam;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_polynomial_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

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
    const std::vector<plssvm::real_type> all_rhos{ plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 } };
    ASSERT_EQ(rho.size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, all_rhos);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());
    EXPECT_EQ(header_lines, 11);
}
TYPED_TEST(LIBSVMModelHeaderParseValid, read_rbf) {
    using label_type = TypeParam;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_rbf_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

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
    const std::vector<plssvm::real_type> all_rhos{ plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 } };
    ASSERT_EQ(rho.size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_VECTOR_EQ(rho, all_rhos);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>().first);
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());
    EXPECT_EQ(header_lines, 9);
}

template <typename T>
class LIBSVMModelHeaderParseInvalid : public LIBSVMModelHeaderParse<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderParseInvalid, util::label_type_gtest , naming::label_type_to_name);

TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_svm_type) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can only use c_svc as svm_type, but 'nu_svc' was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_kernel_type) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized kernel type 'foo'!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, wrong_total_sv) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/wrong_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of support vectors must be greater than 0!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_label) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two labels must be set, but only one label was given!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_nr_sv) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "At least two nr_sv must be set, but only one was given!");
}
//TYPED_TEST(LIBSVMModelHeaderParseInvalid, too_few_rho) {
//    using label_type = TypeParam;
//
//    // parse the LIBSVM model file
//    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_rho.libsvm.model";
//    plssvm::detail::io::file_reader reader{ filename };
//    reader.read_lines('#');
//    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
//                      plssvm::invalid_file_format_exception,
//                      "At least one rho value must be set, but none was given!");
//}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, unrecognized_header_entry) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/unrecognized_header_entry.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry 'invalid entry'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_svm_type) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_svm_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_kernel_type) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_kernel_type.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing kernel_type!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_linear_kernel) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_gamma_in_linear_kernel) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_gamma_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_linear_kernel) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_linear_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_degree_in_rbf_kernel) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_degree_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the degree parameter which is not used in the radial basis function kernel!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, explicit_coef0_in_rbf_kernel) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/explicit_coef0_in_rbf_kernel.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Explicitly provided a value for the coef0 parameter which is not used in the radial basis function kernel!");
}

TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_class) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of different classes nr_class!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_total_sv) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_total_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing total number of support vectors total_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_rho) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_rho.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing rho values!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_label) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_label.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing class label specification!");
}
//TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_label_mismatch) {
//    using label_type = TypeParam;
//
//    // parse the LIBSVM model file
//    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_label_mismatch.libsvm.model";
//    plssvm::detail::io::file_reader reader{ filename };
//    reader.read_lines('#');
//    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
//                      plssvm::invalid_file_format_exception,
//                      "The number of classes (nr_class) is 2, but the provided number of different labels is 3 (label)!");
//}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_nr_sv) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_nr_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing number of support vectors per class nr_sv!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, nr_class_and_nr_sv_mismatch) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/nr_class_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The number of classes (nr_class) is 2, but the provided number of different labels is 3 (nr_sv)!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, total_sv_and_nr_sv_mismatch) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/total_sv_and_nr_sv_mismatch.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The total number of support vectors is 5, but the sum of nr_sv is 6!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_sv) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_sv.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Unrecognized header entry '-1.8568721894e-01 1:-1.1178275006e+00 2:-2.9087188881e+00 3:6.6638344270e-01 4:1.0978832704e+00'! Maybe SV is missing?");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, missing_support_vectors) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_support_vectors.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no support vectors are given or SV is missing!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, same_class_multiple_times) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/same_class_multiple_times.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Provided 2 labels but only 1 of them was/where unique!");
}
TYPED_TEST(LIBSVMModelHeaderParseInvalid, empty) {
    using label_type = TypeParam;

    // parse the LIBSVM model file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Missing svm_type!");
}

template <typename T>
class LIBSVMModelWriteBase : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {};

template <typename T>
class LIBSVMModelHeaderWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWrite, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(LIBSVMModelHeaderWrite, write_linear) {
    using label_type = TypeParam;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_alpha_values = distinct_label.size() == 2 ? 1 : distinct_label.size();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    plssvm::parameter params{};
    params.kernel_type = plssvm::kernel_function_type::linear;
    const std::vector<plssvm::real_type> rho(num_alpha_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

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
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(6), fmt::format("rho {}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(7), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_polynomial) {
    using label_type = TypeParam;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_alpha_values = distinct_label.size() == 2 ? 1 : distinct_label.size();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    plssvm::parameter params{};
    params.kernel_type = plssvm::kernel_function_type::polynomial;
    params.degree = 3;
    params.gamma = 2.2;
    params.coef0 = 4.4;
    const std::vector<plssvm::real_type> rho(num_alpha_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

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
    EXPECT_EQ(reader.line(7), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(8), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(9), fmt::format("rho {}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(10), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_rbf) {
    using label_type = TypeParam;

    // create the output file
    fmt::ostream out = fmt::output_file(this->filename);

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_alpha_values = distinct_label.size() == 2 ? 1 : distinct_label.size();
    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    plssvm::parameter params{};
    params.kernel_type = plssvm::kernel_function_type::rbf;
    params.gamma = 0.4;
    const std::vector<plssvm::real_type> rho(num_alpha_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

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
    EXPECT_EQ(reader.line(5), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(6), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(7), fmt::format("rho {}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(8), "SV");
}

template <typename T>
class LIBSVMModelHeaderWriteDeathTest : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWriteDeathTest, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(LIBSVMModelHeaderWriteDeathTest, write_header_without_label) {
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> rho(util::get_num_classes<label_type>());
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<plssvm::real_type>>{ { plssvm::real_type{ 0.0 } } } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelHeaderWriteDeathTest, write_header_invalid_number_of_rho_values) {
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> rho{};
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<plssvm::real_type>>{ { plssvm::real_type{ 0.0 } } },
                                                 std::vector<label_type>{ util::get_distinct_label<label_type>().front() } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set)),
                 ::testing::HasSubstr("At least one rho value must be provided!"));
}

template <typename T>
class LIBSVMModelDataParseDense : public ::testing::Test, protected util::temporary_file {
  protected:
    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        const std::string template_filename = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
        util::instantiate_template_file<label_type>(template_filename, this->filename);
    }

    using real_type = plssvm::real_type;
    using label_type = T;

    const std::vector<std::vector<real_type>> correct_data{
        { real_type{ -1.1178275006 }, real_type{ -2.9087188881 }, real_type{ 0.66638344270 }, real_type{ 1.0978832704 } },
        { real_type{ -0.52821182989 }, real_type{ -0.33588098497 }, real_type{ 0.51687296030 }, real_type{ 0.54604461446 } },
        { real_type{ 0.57650218263 }, real_type{ 1.0140559662 }, real_type{ 0.13009428080 }, real_type{ 0.72619138869 } },
        { real_type{ 1.8849404372 }, real_type{ 1.0051856432 }, real_type{ 0.29849993305 }, real_type{ 1.6464627049 } },
        { real_type{ -0.20981208921 }, real_type{ 0.60276937379 }, real_type{ -0.13086851759 }, real_type{ 0.10805254527 } },
        { real_type{ -1.1256816276 }, real_type{ 2.1254153434 }, real_type{ -0.16512657655 }, real_type{ 2.5164553141 } }
    };
    const std::vector<std::vector<real_type>> correct_weights{
        { real_type{ -1.8568721894e-01 }, real_type{ 9.0116552290e-01 }, real_type{ -2.2483112395e-01 }, real_type{ 1.4909749921e-02 }, real_type{ -4.5666857706e-01 }, real_type{ -4.8888352876e-02 } },
        { real_type{ 1.1365048527e-01 }, real_type{ -3.2357185930e-01 }, real_type{ 8.9871548758e-01 }, real_type{ -7.5259922896e-02 }, real_type{ -4.7955922738e-01 }, real_type{ -1.3397496327e-01 } },
        { real_type{ 2.8929914669e-02 }, real_type{ -4.8559849173e-01 }, real_type{ -5.6740083618e-01 }, real_type{ 8.7841608802e-02 }, real_type{ 9.7960957282e-01 }, real_type{ -4.3381768383e-02 } },
        { real_type{ 4.3106819001e-02 }, real_type{ -9.1995171877e-02 }, real_type{ -1.0648352745e-01 }, real_type{ -2.7491435827e-02 }, real_type{ -4.3381768383e-02 }, real_type{ 2.2624508453e-01 } }
    };
};
TYPED_TEST_SUITE(LIBSVMModelDataParseDense, util::label_type_gtest, naming::label_type_to_name);

class LIBSVMModelDataParse : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMModelDataParse, util::label_type_gtest, naming::label_type_to_name);

//TYPED_TEST(LIBSVMModelDataParseDense, read) {
//    using current_label_type = TypeParam;
//
//    // parse the LIBSVM file
//    plssvm::detail::io::file_reader reader{ this->filename };
//    reader.read_lines('#');
//    // skip the first 8 lines, i.e., the model file header using the linear kernel function
//    const std::size_t num_classes_for_label_type = util::get_num_classes<current_label_type>();
//    const auto num_sv_per_class = util::detail::calculate_model_file_label_distribution<current_label_type>(num_classes_for_label_type);
//    const auto [num_data_points, num_features, data, alpha, classification] = plssvm::detail::io::parse_libsvm_model_data(reader, num_sv_per_class, 8);
//
//    // check for correct sizes
//    ASSERT_EQ(num_data_points, 6);
//    ASSERT_EQ(num_features, 4);
//    EXPECT_EQ(classification, plssvm::classification_type::oao);
//
//    // check for correct data
//    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, decltype(data){ this->correct_data });
//
//    // check for correct weights
//    ASSERT_EQ(alpha.size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
//
//    std::vector<std::vector<std::vector<plssvm::real_type>>> weights;
//    switch (num_classes_for_label_type) {
//        case 2:
//            weights.emplace_back(std::vector<std::vector<plssvm::real_type>>{ this->correct_weights[0] });
//            break;
//        case 3:
//            weights.emplace_back(std::vector<std::vector<plssvm::real_type>>{ this->correct_weights[0], this->correct_weights[1], this->correct_weights[2] });
//            break;
//        case 4:
//            weights.emplace_back(std::vector<std::vector<plssvm::real_type>>{ this->correct_weights });
//            break;
//        default:
//            FAIL() << "Unreachable!";
//            break;
//    }
//
//    ASSERT_EQ(alpha.size(), weights.size());
//    for (std::size_t i = 0; i < alpha.size(); ++i) {
//        EXPECT_FLOATING_POINT_MATRIX_NEAR(alpha[i], plssvm::aos_matrix<plssvm::real_type>{ weights[i] });
//    }
//}


TEST(LIBSVMModelDataParse, zero_based_features) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/zero_based_features.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}
TEST(LIBSVMModelDataParse, empty) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/empty.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no data points are given!");
}
TEST(LIBSVMModelDataParse, too_few_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: need 1 alpha values, but only 0 were given!");
}
TEST(LIBSVMModelDataParse, too_many_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_many_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: too many alpha values were given!");
}
TEST(LIBSVMModelDataParse, feature_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/feature_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert 'a-1.1178275006e+00' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TEST(LIBSVMModelDataParse, index_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/index_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' !2' to a value of type unsigned long!");
}
TEST(LIBSVMModelDataParse, invalid_colon_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/invalid_colon_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: need 1 alpha values, but only 0 were given!");
}
TEST(LIBSVMModelDataParse, invalid_colon_in_the_middle) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/invalid_colon_in_the_middle.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}
TEST(LIBSVMModelDataParse, missing_feature_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_feature_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TEST(LIBSVMModelDataParse, missing_index_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_index_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}
TEST(LIBSVMModelDataParse, non_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/non_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 3 is smaller or equal than 3!");
}
TEST(LIBSVMModelDataParse, non_strictly_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/non_strictly_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 2 is smaller or equal than 3!");
}

template <typename T>
class LIBSVMModelDataParseDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMModelDataParseDeathTest, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(LIBSVMModelDataParseDeathTest, invalid_file_reader) {
    // open file_reader without associating it to a file
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 0)),
                 "The file_reader is currently not associated with a file!");
}
//TYPED_TEST(LIBSVMModelDataParseDeathTest, too_few_alpha_values) {
//    // parse LIBSVM file
//    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm";
//    plssvm::detail::io::file_reader reader{ filename };
//    reader.read_lines('#');
//    // try to skip more lines than are present in the data file
//    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 0)),
//                 "At least one alpha value must be present!");
//}
//TYPED_TEST(LIBSVMModelDataParseDeathTest, two_alpha_values) {
//    // parse LIBSVM file
//    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm";
//    plssvm::detail::io::file_reader reader{ filename };
//    reader.read_lines('#');
//    // try to skip more lines than are present in the data file
//    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 0)),
//                 ::testing::HasSubstr("Two alpha values may never be present (binary classification as special case only uses 1 alpha value)!"));
//}
TYPED_TEST(LIBSVMModelDataParseDeathTest, skip_too_many_lines) {
    // parse LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/6x4_linear.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // try to skip more lines than are present in the data file
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 15)),
                 "Tried to skipp 15 lines, but only 14 are present!");
}

template <typename T>
class LIBSVMModelDataWrite : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelDataWrite, util::label_type_gtest , naming::label_type_to_name);

//TYPED_TEST(LIBSVMModelDataWrite, write) {
//    using real_type = plssvm::real_type;
//    using label_type = TypeParam;
//
//    // define data to write
//    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
//    const std::size_t num_alpha_values = distinct_label.size() == 2 ? 1 : distinct_label.size();
//    const auto [label, num_sv] = util::get_correct_model_file_labels<label_type>();
//    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<real_type>>(label.size(), 3);
//
//    // create necessary parameter
//    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
//    const std::vector<real_type> rho(num_alpha_values, real_type{ 3.1415 });
//    const auto alpha = util::generate_specific_matrix<plssvm::aos_matrix<real_type>>(num_alpha_values, data.num_rows());
//    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };
//
//    // write the LIBSVM model file
//    plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set);
//
//    // read the written file
//    plssvm::detail::io::file_reader reader{ this->filename };
//    reader.read_lines('#');
//
//    // check the written data
//    ASSERT_EQ(reader.num_lines(), 8 + data.num_rows());  // header lines + data
//    EXPECT_EQ(reader.line(0), "svm_type c_svc");
//    EXPECT_EQ(reader.line(1), "kernel_type linear");
//    EXPECT_EQ(reader.line(2), fmt::format("nr_class {}", distinct_label.size()));
//    EXPECT_EQ(reader.line(3), fmt::format("label {}", fmt::join(distinct_label, " ")));
//    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
//    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
//    EXPECT_EQ(reader.line(6), fmt::format("rho {}", fmt::join(rho, " ")));
//    EXPECT_EQ(reader.line(7), "SV");
//
//    // sort data points into buckets for each label type
//    std::map<label_type, std::vector<std::pair<std::vector<real_type>, std::vector<real_type>>>> data_per_label;
//    for (std::size_t i = 0; i < label.size(); ++i) {
//        if (!plssvm::detail::contains(data_per_label, label[i])) {
//            data_per_label[label[i]] = std::vector<std::pair<std::vector<real_type>, std::vector<real_type>>>{};
//        }
//        std::vector<real_type> alpha_val;
//        for (std::size_t j = 0; j < alpha.size(); ++j) {
//            alpha_val.emplace_back(alpha[j][i]);
//        }
//        data_per_label[label[i]].emplace_back(std::move(alpha_val), data[i]);
//    }
//
//    // check lines for correctness
//    std::size_t idx = 8;
//    for (std::size_t i = 0; i < distinct_label.size(); ++i) {
//        for (std::size_t j = 0; j < num_sv[i]; ++j) {
//            // check if any of the data would match the current line
//            bool line_found = false;
//            for (const auto &[alpha_val, vec] : data_per_label[distinct_label[i]]) {
//                const std::string line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", fmt::join(alpha_val, " "), vec[0], vec[1], vec[2]);
//                if (reader.line(idx) == line) {
//                    line_found = true;
//                }
//            }
//            if (!line_found) {
//                GTEST_FAIL() << fmt::format("Couldn't find the line '{}' ({}) from the output file in the provided data set.", reader.line(idx), idx);
//            }
//            ++idx;
//        }
//
//    }
//}

template <typename T>
class LIBSVMModelWriteDeathTest : public LIBSVMModelWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMModelWriteDeathTest, util::label_type_gtest, naming::label_type_to_name);

TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_without_label) {
    using real_type = plssvm::real_type;
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::size_t num_classes = util::get_num_classes<label_type>();
    const std::size_t num_alpha_values = num_classes == 2 ? 1 : num_classes;
    const std::vector<real_type> rho(num_alpha_values);
    const std::vector<std::vector<real_type>> alpha(num_alpha_values, std::vector<real_type>{ real_type{ 0.1 } });
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<real_type>>{ { real_type{ 0.0 } } } };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_invalid_number_of_rho_values) {
    using real_type = plssvm::real_type;
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::vector<real_type> rho{};
    const auto alpha =  util::generate_random_matrix<plssvm::aos_matrix<real_type>>(util::get_num_classes<label_type>(), 1);
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<real_type>>{ { real_type{ 0.0 } } }, std::vector<label_type>{ util::get_distinct_label<label_type>().front() } };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 ::testing::HasSubstr("The number of rho values (0) must be equal to the number of different labels (1) or must be 1 for binary classification!"));
}
TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_invalid_number_of_weights) {
    using real_type = plssvm::real_type;
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::size_t num_classes = util::get_num_classes<label_type>();
    const std::size_t num_alpha_values = num_classes == 2 ? 1 : num_classes;
    const std::vector<real_type> rho(num_alpha_values);
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>().first;
    const auto alpha = util::generate_random_matrix<plssvm::aos_matrix<real_type>>(num_classes + 1, label.size());
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<real_type>>(label.size(), 1);

    const plssvm::data_set<label_type> data_set{ data, std::move(label) };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 ::testing::HasSubstr(fmt::format("The number of weight vectors ({}) must be equal to the number of different labels ({}) or must be 1 for binary classification!", num_classes + 1, num_classes)));
}
TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_too_few_weights_in_class) {
    using real_type = plssvm::real_type;
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::size_t num_classes = util::get_num_classes<label_type>();
    const std::size_t num_alpha_values = num_classes == 2 ? 1 : num_classes;
    const std::vector<real_type> rho(num_alpha_values);
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>().first;
    auto alpha{ util::generate_random_matrix<plssvm::aos_matrix<real_type>>(num_alpha_values, label.size()) };
//    alpha.front().pop_back();  // TODO:
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<real_type>>(label.size(), 1);

    const plssvm::data_set<label_type> data_set{ data, std::move(label) };

    if (num_classes == 2) {
        // binary classification
        // try writing the LIBSVM model header
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                     ::testing::HasSubstr(fmt::format("The number of weights (5) must be equal to the number of support vectors (6)!")));
    } else {
        // try writing the LIBSVM model header
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                     ::testing::HasSubstr(fmt::format("The number of weights per class must be equal!")));
    }
}
TYPED_TEST(LIBSVMModelWriteDeathTest, write_data_invalid_number_of_weights_per_class) {
    using real_type = plssvm::real_type;
    using label_type = TypeParam;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::size_t num_classes = util::get_num_classes<label_type>();
    const std::size_t num_alpha_values = num_classes == 2 ? 1 : num_classes;
    const std::vector<real_type> rho(num_alpha_values);
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>().first;
    const auto alpha = util::generate_random_matrix<plssvm::aos_matrix<real_type>>(num_alpha_values, label.size() - 1);
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<real_type>>(label.size(), 1);

    const plssvm::data_set<label_type> data_set{ data, std::move(label) };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, params, rho, alpha, data_set)),
                 ::testing::HasSubstr(fmt::format("The number of weights (5) must be equal to the number of support vectors (6)!")));
}