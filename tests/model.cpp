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
#include "utility.hpp"             // util::{temporary_file, redirect_output, instantiate_template_file, get_distinct_label}

#include "gtest/gtest.h"           // EXPECT_EQ, EXPECT_TRUE, ASSERT_GT, GTEST_FAIL, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                                   // ::testing::{StaticAssertTypeEq, Test, TestWithParam, Values}

#include <cstddef>                 // std::size_t
#include <regex>                   // std::regex, std::regex_match, std::regex::extended
#include <string>                  // std::string
#include <string_view>             // std::string_view
#include <vector>                  // std::vector

template <typename T>
class Model : public ::testing::Test, private util::redirect_output<> {};
TYPED_TEST_SUITE(Model, util::real_type_label_type_combination_gtest, naming::real_type_label_type_combination_to_name);

TYPED_TEST(Model, typedefs) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
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
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for correct construction
    EXPECT_EQ(model.num_support_vectors(), 5);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params(), plssvm::parameter{});
    const std::vector<std::vector<real_type>> support_vectors{
        { real_type{ -1.117828 }, real_type{ -2.908719 }, real_type{ 0.6663834 }, real_type{ 1.097883 } },
        { real_type{ -0.5282118 }, real_type{ -0.3358810 }, real_type{ 0.5168730 }, real_type{ 0.5460446 } },
        { real_type{ -0.2098121 }, real_type{ 0.6027694 }, real_type{ -0.1308685 }, real_type{ 0.1080525 } },
        { real_type{ 1.884940 }, real_type{ 1.005186 }, real_type{ 0.2984999 }, real_type{ 1.646463 } },
        { real_type{ 0.5765022 }, real_type{ 1.014056 }, real_type{ 0.1300943 }, real_type{ 0.7261914 } }
    };
    EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.support_vectors(), support_vectors);
    const std::vector<real_type> weights{
        real_type{ -0.17609610490769723 }, real_type{ 0.8838187731213127 }, real_type{ -0.47971257671001616 }, real_type{ 0.0034556484621847128 }, real_type{ -0.23146573996578407 }
    };
    EXPECT_FLOATING_POINT_VECTOR_EQ(model.weights(), weights);
    EXPECT_FLOATING_POINT_EQ(model.rho(), real_type{ 0.37330625882191915 });
}

TYPED_TEST(Model, num_support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of support vectors
    EXPECT_EQ(model.num_support_vectors(), 5);
}
TYPED_TEST(Model, num_features) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of features
    EXPECT_EQ(model.num_features(), 4);
}
TYPED_TEST(Model, get_params) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct number of features
    EXPECT_EQ(model.get_params(), plssvm::parameter{});
}
TYPED_TEST(Model, support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct support vectors
    const std::vector<std::vector<real_type>> support_vectors{
        { real_type{ -1.117828 }, real_type{ -2.908719 }, real_type{ 0.6663834 }, real_type{ 1.097883 } },
        { real_type{ -0.5282118 }, real_type{ -0.3358810 }, real_type{ 0.5168730 }, real_type{ 0.5460446 } },
        { real_type{ -0.2098121 }, real_type{ 0.6027694 }, real_type{ -0.1308685 }, real_type{ 0.1080525 } },
        { real_type{ 1.884940 }, real_type{ 1.005186 }, real_type{ 0.2984999 }, real_type{ 1.646463 } },
        { real_type{ 0.5765022 }, real_type{ 1.014056 }, real_type{ 0.1300943 }, real_type{ 0.7261914 } }
    };
    EXPECT_FLOATING_POINT_2D_VECTOR_EQ(model.support_vectors(), support_vectors);
}
TYPED_TEST(Model, labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // get the distinct labels used to instantiate the test file
    const std::pair<label_type, label_type> distinct_label = util::get_distinct_label<label_type>();

    // correct labels
    const std::vector<label_type> correct_label = { distinct_label.first, distinct_label.first, distinct_label.second, distinct_label.second, distinct_label.second };

    // check labels getter
    EXPECT_EQ(model.labels(), correct_label);
}
TYPED_TEST(Model, num_different_labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // check num_different_labels getter
    EXPECT_EQ(model.num_different_labels(), 2);
}
 TYPED_TEST(Model, different_labels) {
     using real_type = typename TypeParam::real_type;
     using label_type = typename TypeParam::label_type;

     // instantiate a model file
     const util::temporary_file model_file;
     util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
     const plssvm::model<real_type, label_type> model{ model_file.filename };

     // get the distinct labels used to instantiate the test file
     const std::pair<label_type, label_type> distinct_label = util::get_distinct_label<label_type>();
     const std::vector<label_type> different_label = { distinct_label.first, distinct_label.second};

     // check different_labels getter
     EXPECT_EQ(model.different_labels(), different_label);
 }
TYPED_TEST(Model, weights) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct weights
    const std::vector<real_type> weights{
        real_type{ -0.17609610490769723 }, real_type{ 0.8838187731213127 }, real_type{ -0.47971257671001616 }, real_type{ 0.0034556484621847128 }, real_type{ -0.23146573996578407 }
    };
    EXPECT_FLOATING_POINT_VECTOR_EQ(model.weights(), weights);
}
TYPED_TEST(Model, rho) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // instantiate a model file
    const util::temporary_file model_file;
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/5x4_linear_TEMPLATE.libsvm.model", model_file.filename);
    const plssvm::model<real_type, label_type> model{ model_file.filename };

    // test for the correct rho (bias) value
    EXPECT_FLOATING_POINT_EQ(model.rho(), real_type{ 0.37330625882191915 });
}

class ModelSave : public ::testing::TestWithParam<std::string>, private util::redirect_output<>, protected util::temporary_file {};
TEST_P(ModelSave, save) {
    // create a model using an existing LIBSVM model file
    const plssvm::model<double, int> model{ fmt::format("{}{}", PLSSVM_TEST_PATH, GetParam()) };

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
    regex_patterns.emplace_back("label (.+ ?)+");
    regex_patterns.emplace_back("nr_sv ([0-9]+ ?)+");
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
                "/data/model/5x4_linear.libsvm.model",
                "/data/model/5x4_polynomial.libsvm.model",
                "/data/model/5x4_rbf.libsvm.model"));
// clang-format on