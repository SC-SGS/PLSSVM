/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the model class representing a learned SVM model.
 */

#include "plssvm/model.hpp"

#include "plssvm/parameter.hpp"    // plssvm::parameter

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ, EXPECT_FLOATING_POINT_VECTOR_EQ, EXPECT_FLOATING_POINT_2D_VECTOR_EQ
#include "naming.hpp"              // naming::real_type_label_type_combination_to_name
#include "types_to_test.hpp"       // util::real_type_label_type_combination_gtest
#include "utility.hpp"             // util::{temporary_file, redirect_output, instantiate_template_file, get_distinct_label, get_correct_model_file_labels}

#include "gtest/gtest.h"           // EXPECT_EQ, EXPECT_TRUE, ASSERT_GT, GTEST_FAIL, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                                   // ::testing::{StaticAssertTypeEq, Test, TestWithParam, Values}

#include <cstddef>                 // std::size_t
#include <regex>                   // std::regex, std::regex_match, std::regex::extended
#include <string>                  // std::string
#include <string_view>             // std::string_view
#include <tuple>                   // std::tuple
#include <vector>                  // std::vector

template <typename T>
class Model : public ::testing::Test, private util::redirect_output<> {};
TYPED_TEST_SUITE(Model, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(Model, typedefs) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<real_type, typename decltype(model)::real_type>();
    ::testing::StaticAssertTypeEq<label_type, typename decltype(model)::label_type>();
}

TYPED_TEST(Model, construct) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for correct construction
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params(), plssvm::parameter{});
    const std::vector<std::vector<real_type>> support_vectors{
        { real_type{ -1.1178275006 }, real_type{ -2.9087188881 }, real_type{ 0.66638344270 }, real_type{ 1.0978832704 } },
        { real_type{ -0.52821182989 }, real_type{ -0.33588098497 }, real_type{ 0.51687296030 }, real_type{ 0.54604461446 } },
        { real_type{ 0.57650218263 }, real_type{ 1.0140559662 }, real_type{ 0.13009428080 }, real_type{ 0.72619138869 } },
        { real_type{ 1.8849404372 }, real_type{ 1.0051856432 }, real_type{ 0.29849993305 }, real_type{ 1.6464627049 } },
        { real_type{ -0.20981208921 }, real_type{ 0.60276937379 }, real_type{ -0.13086851759 }, real_type{ 0.10805254527 } },
        { real_type{ -1.1256816276 }, real_type{ 2.1254153434 }, real_type{ -0.16512657655 }, real_type{ 2.5164553141 } }
    };
    EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.support_vectors(), support_vectors);

    const std::vector<std::vector<real_type>> all_weights{
        { real_type{ -1.8568721894e-01 }, real_type{ 9.0116552290e-01 }, real_type{ -2.2483112395e-01 }, real_type{ 1.4909749921e-02 }, real_type{ -4.5666857706e-01 }, real_type{ -4.8888352876e-02 } },
        { real_type{ 1.1365048527e-01 }, real_type{ -3.2357185930e-01 }, real_type{ 8.9871548758e-01 }, real_type{ -7.5259922896e-02 }, real_type{ -4.7955922738e-01 }, real_type{ -1.3397496327e-01 } },
        { real_type{ 2.8929914669e-02 }, real_type{ -4.8559849173e-01 }, real_type{ -5.6740083618e-01 }, real_type{ 8.7841608802e-02 }, real_type{ 9.7960957282e-01 }, real_type{ -4.3381768383e-02 } },
        { real_type{ 4.3106819001e-02 }, real_type{ -9.1995171877e-02 }, real_type{ -1.0648352745e-01 }, real_type{ -2.7491435827e-02 }, real_type{ -4.3381768383e-02 }, real_type{ 2.2624508453e-01 } }
    };
    const std::vector<real_type> all_rhos{ real_type{ 0.32260160011873423 }, real_type{ 0.401642656885171 }, real_type{ 0.05160647594201395 }, real_type{ 1.224149267054074 } };

    ASSERT_EQ(model.weights().size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), (std::vector<std::vector<real_type>>{ all_weights[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), (std::vector<std::vector<real_type>>{ all_weights[0], all_weights[1], all_weights[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), all_weights);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }

    ASSERT_EQ(model.rho().size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<real_type>{ all_rhos[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), all_rhos);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
}

TYPED_TEST(Model, num_support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of support vectors
    EXPECT_EQ(model.num_support_vectors(), 6);
}
TYPED_TEST(Model, num_features) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of features
    EXPECT_EQ(model.num_features(), 4);
}
TYPED_TEST(Model, get_params) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of features
    EXPECT_EQ(model.get_params(), plssvm::parameter{});
}
TYPED_TEST(Model, support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct support vectors
    const std::vector<std::vector<real_type>> support_vectors{
        { real_type{ -1.1178275006 }, real_type{ -2.9087188881 }, real_type{ 0.66638344270 }, real_type{ 1.0978832704 } },
        { real_type{ -0.52821182989 }, real_type{ -0.33588098497 }, real_type{ 0.51687296030 }, real_type{ 0.54604461446 } },
        { real_type{ 0.57650218263 }, real_type{ 1.0140559662 }, real_type{ 0.13009428080 }, real_type{ 0.72619138869 } },
        { real_type{ 1.8849404372 }, real_type{ 1.0051856432 }, real_type{ 0.29849993305 }, real_type{ 1.6464627049 } },
        { real_type{ -0.20981208921 }, real_type{ 0.60276937379 }, real_type{ -0.13086851759 }, real_type{ 0.10805254527 } },
        { real_type{ -1.1256816276 }, real_type{ 2.1254153434 }, real_type{ -0.16512657655 }, real_type{ 2.5164553141 } }
    };
    EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.support_vectors(), support_vectors);
}
TYPED_TEST(Model, labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // check labels getter
    EXPECT_EQ(model.labels(), util::get_correct_model_file_labels<label_type>().first);
}
TYPED_TEST(Model, num_different_labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // check num_different_labels getter
    EXPECT_EQ(model.num_different_labels(), util::get_num_classes<label_type>());
}
 TYPED_TEST(Model, different_labels) {
     using real_type = typename TypeParam::real_type;
     using label_type = typename TypeParam::label_type;

     // instantiate a model file
     const util::temporary_file model_file;
     const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", util::get_num_classes<label_type>());
     util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
     const plssvm::model<real_type, label_type> model{ model_file.filename };

     // check different_labels getter
     EXPECT_EQ(model.different_labels(), util::get_distinct_label<label_type>());
 }
TYPED_TEST(Model, weights) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct weights
    const std::vector<std::vector<real_type>> all_weights{
        { real_type{ -1.8568721894e-01 }, real_type{ 9.0116552290e-01 }, real_type{ -2.2483112395e-01 }, real_type{ 1.4909749921e-02 }, real_type{ -4.5666857706e-01 }, real_type{ -4.8888352876e-02 } },
        { real_type{ 1.1365048527e-01 }, real_type{ -3.2357185930e-01 }, real_type{ 8.9871548758e-01 }, real_type{ -7.5259922896e-02 }, real_type{ -4.7955922738e-01 }, real_type{ -1.3397496327e-01 } },
        { real_type{ 2.8929914669e-02 }, real_type{ -4.8559849173e-01 }, real_type{ -5.6740083618e-01 }, real_type{ 8.7841608802e-02 }, real_type{ 9.7960957282e-01 }, real_type{ -4.3381768383e-02 } },
        { real_type{ 4.3106819001e-02 }, real_type{ -9.1995171877e-02 }, real_type{ -1.0648352745e-01 }, real_type{ -2.7491435827e-02 }, real_type{ -4.3381768383e-02 }, real_type{ 2.2624508453e-01 } }
    };

    ASSERT_EQ(model.weights().size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), (std::vector<std::vector<real_type>>{ all_weights[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), (std::vector<std::vector<real_type>>{ all_weights[0], all_weights[1], all_weights[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.weights(), all_weights);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
}
TYPED_TEST(Model, rho) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_TEMPLATE.libsvm.model", num_classes_for_label_type);
    util::instantiate_template_file<label_type>(template_file_name, model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct rho (bias) value
    const std::vector<real_type> all_rhos{ real_type{ 0.32260160011873423 }, real_type{ 0.401642656885171 }, real_type{ 0.05160647594201395 }, real_type{ 1.224149267054074 } };

    ASSERT_EQ(model.rho().size(), num_classes_for_label_type == 2 ? 1 : num_classes_for_label_type);
    switch (num_classes_for_label_type) {
        case 2:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<real_type>{ all_rhos[0] }));
            break;
        case 3:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
            break;
        case 4:
            EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), all_rhos);
            break;
        default:
            FAIL() << "Unreachable!";
            break;
    }
}

class ModelSave : public ::testing::TestWithParam<std::string>, private util::redirect_output<>, protected util::temporary_file {};
TEST_P(ModelSave, save) {
    // create a model using an existing LIBSVM model file
    const plssvm::model<double, int> model{ fmt::format("{}{}", PLSSVM_TEST_PATH, GetParam()) };
    // note: binary classification hardcoded -> only two alpha values provided

    // write model to file
    model.save(filename);

    // read previously written file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // copy read lines
    std::vector<std::string_view> lines{ reader.lines() };

    // create vector containing correct regex expressions for the LIBSVM model file header
    std::vector<std::string> regex_patterns;
    regex_patterns.emplace_back("svm_type c_svc");
    regex_patterns.emplace_back(fmt::format("kernel_type {}", model.get_params().kernel_type));
    switch (model.get_params().kernel_type) {
        case plssvm::kernel_function_type::linear:
            break;
        case plssvm::kernel_function_type::polynomial:
            regex_patterns.emplace_back("degree [0-9]+");
            regex_patterns.emplace_back("gamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
            regex_patterns.emplace_back("coef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
            break;
        case plssvm::kernel_function_type::rbf:
            regex_patterns.emplace_back("gamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
            break;
    }
    regex_patterns.emplace_back("nr_class [0-9]+");
    regex_patterns.emplace_back("total_sv [0-9]+");
    regex_patterns.emplace_back("rho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
    regex_patterns.emplace_back("label .+ .+");
    regex_patterns.emplace_back("nr_sv [0-9]+ [0-9]+");
    regex_patterns.emplace_back("SV");

    // at least number of header entries lines must be present
    ASSERT_GT(reader.num_lines(), regex_patterns.size());

    // check if the model header is valid
    for (const std::string &pattern : regex_patterns) {
        const std::regex reg{ pattern, std::regex::extended };

        // check each line if one matches the regex pattern
        bool found_matching_line{ false };
        for (std::size_t i = 0; i < lines.size(); ++i) {
            // check if ANY line matches the current regex pattern
            if (std::regex_match(std::string{ lines[i] }, reg)) {
                found_matching_line = true;
                // remove this line since it already matched a regex pattern
                lines.erase(lines.begin() + static_cast<std::vector<std::string_view>::iterator::difference_type>(i));
                break;
            }
        }
        // NO line matches the pattern -> test failed
        if (!found_matching_line) {
            GTEST_FAIL() << fmt::format(R"(Can't find a line matching the regex pattern: "{}".)", pattern);
        }
    }
    // only support vectors should be left -> check the remaining lines if they match the correct pattern
    const std::string support_vector_pattern{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?)*" };
    for (const std::string_view line : lines) {
        const std::regex reg(support_vector_pattern, std::regex::extended);
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg)) << fmt::format(R"(Line "{}" doesn't match the regex pattern "{}".)", line, support_vector_pattern);
    }
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(Model, ModelSave, ::testing::Values(
                "/data/model/6x4_linear.libsvm.model",
                "/data/model/6x4_polynomial.libsvm.model",
                "/data/model/6x4_rbf.libsvm.model"));
// clang-format on