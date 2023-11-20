/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing an invalid LIBSVM model file header.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/classification_types.hpp"   // plssvm::classification_type
#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_VECTOR_EQ, EXPECT_FLOATING_POINT_EQ
#include "naming.hpp"              // naming::parameter_definition_to_name
#include "types_to_test.hpp"       // util::label_type_classification_type_gtest
#include "utility.hpp"             // util::{temporary_file, get_num_classes, instantiate_template_file, get_correct_model_file_labels,
                                   // get_distinct_label, get_correct_model_file_num_sv_per_class}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ASSERT_EQ, FAIL, FAIL, ::testing::Test

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

template <typename T>
class LIBSVMModelHeaderParseValid : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;
};
TYPED_TEST_SUITE(LIBSVMModelHeaderParseValid, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelHeaderParseValid, read_linear) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_{}_TEMPLATE.libsvm.model", num_classes_for_label_type, classification);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, num_header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

    // check for correctness

    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::linear);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_TRUE(params.gamma.is_default());
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());

    // check rho values
    const std::vector<plssvm::real_type> all_rhos{
        plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 }, plssvm::real_type{ -0.2415331131484474 }, plssvm::real_type{ -2.636779683484747e-16 }
    };

    if constexpr (classification == plssvm::classification_type::oaa) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2], all_rhos[3] }));
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
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
        }
    } else {
        FAIL() << "unknown classification type";
    }

    // check labels
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>());

    // check different classes
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());

    // check number of support vectors per class
    EXPECT_EQ(num_sv_per_class, util::get_correct_model_file_num_sv_per_class<label_type>(label.size()));

    // check number of header lines
    EXPECT_EQ(num_header_lines, 8);
}

TYPED_TEST(LIBSVMModelHeaderParseValid, read_polynomial) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_polynomial_{}_TEMPLATE.libsvm.model", num_classes_for_label_type, classification);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, num_header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

    // check for correctness

    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_FALSE(params.degree.is_default());
    EXPECT_EQ(params.degree.value(), 2);
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(params.gamma.value(), plssvm::real_type{ 0.25 });
    EXPECT_FALSE(params.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(params.coef0.value(), plssvm::real_type{ 1.5 });
    EXPECT_TRUE(params.cost.is_default());

    // check rho values
    const std::vector<plssvm::real_type> all_rhos{
        plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 }, plssvm::real_type{ -0.2415331131484474 }, plssvm::real_type{ -2.636779683484747e-16 }
    };

    if constexpr (classification == plssvm::classification_type::oaa) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2], all_rhos[3] }));
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
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
        }
    } else {
        FAIL() << "unknown classification type";
    }

    // check labels
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>());

    // check different classes
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());

    // check number of support vectors per class
    EXPECT_EQ(num_sv_per_class, util::get_correct_model_file_num_sv_per_class<label_type>(label.size()));

    // check number of header lines
    EXPECT_EQ(num_header_lines, 11);
}

TYPED_TEST(LIBSVMModelHeaderParseValid, read_rbf) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_rbf_{}_TEMPLATE.libsvm.model", num_classes_for_label_type, classification);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, num_header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

    // check for correctness

    // check parameter
    EXPECT_FALSE(params.kernel_type.is_default());
    EXPECT_EQ(params.kernel_type.value(), plssvm::kernel_function_type::rbf);
    EXPECT_TRUE(params.degree.is_default());
    EXPECT_FALSE(params.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(params.gamma.value(), plssvm::real_type{ 0.025 });
    EXPECT_TRUE(params.coef0.is_default());
    EXPECT_TRUE(params.cost.is_default());

    // check rho values
    const std::vector<plssvm::real_type> all_rhos{
        plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 }, plssvm::real_type{ -0.2415331131484474 }, plssvm::real_type{ -2.636779683484747e-16 }
    };

    if constexpr (classification == plssvm::classification_type::oaa) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2], all_rhos[3] }));
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
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
        }
    } else {
        FAIL() << "unknown classification type";
    }

    // check labels
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>());

    // check different classes
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());

    // check number of support vectors per class
    EXPECT_EQ(num_sv_per_class, util::get_correct_model_file_num_sv_per_class<label_type>(label.size()));

    // check number of header lines
    EXPECT_EQ(num_header_lines, 9);
}