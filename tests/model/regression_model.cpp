/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the regression model class representing a learned SVR model.
 */

#include "plssvm/model/regression_model.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/shape.hpp"                  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_EQ
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{regression_label_type_classification_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_distinct_label, get_correct_model_file_labels}

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, ASSERT_GT, FAIL,
                          // ::testing::{Test, StaticAssertTypeEq}

#include <cstddef>      // std::size_t
#include <regex>        // std::regex, std::regex_match, std::regex::extended
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

template <typename T>
class RegressionModel : public ::testing::Test,
                        private util::redirect_output<>,
                        protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/model/regression/6x4_TEMPLATE.libsvm.model", this->filename);
    }
};

TYPED_TEST_SUITE(RegressionModel, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(RegressionModel, Typedefs) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<label_type, typename decltype(model)::label_type>();
    ::testing::StaticAssertTypeEq<std::size_t, typename decltype(model)::size_type>();
}

TYPED_TEST(RegressionModel, Construct) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for correct construction
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params(), plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::linear });
    EXPECT_EQ(model.support_vectors().shape(), (plssvm::shape{ 6, 4 }));
    ASSERT_FALSE(model.labels().has_value());
    EXPECT_EQ(model.weights().size(), 1);
    EXPECT_EQ(model.weights().front().shape(), (plssvm::shape{ 1, 6 }));
    EXPECT_EQ(model.rho().size(), 1);
    EXPECT_FALSE(model.num_iters().has_value());
}

TYPED_TEST(RegressionModel, NumSupportVectors) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct number of support vectors
    EXPECT_EQ(model.num_support_vectors(), 6);
}

TYPED_TEST(RegressionModel, NumFeatures) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct number of features
    EXPECT_EQ(model.num_features(), 4);
}

TYPED_TEST(RegressionModel, GetParams) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct number of features
    EXPECT_EQ(model.get_params(), plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::linear });
}

TYPED_TEST(RegressionModel, SupportVectors) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct support vectors
    const plssvm::soa_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ -1.1178275006 }, plssvm::real_type{ -2.9087188881 }, plssvm::real_type{ 0.66638344270 }, plssvm::real_type{ 1.0978832704 } },
                                                                   { plssvm::real_type{ -0.52821182989 }, plssvm::real_type{ -0.33588098497 }, plssvm::real_type{ 0.51687296030 }, plssvm::real_type{ 0.54604461446 } },
                                                                   { plssvm::real_type{ 0.57650218263 }, plssvm::real_type{ 1.0140559662 }, plssvm::real_type{ 0.13009428080 }, plssvm::real_type{ 0.72619138869 } },
                                                                   { plssvm::real_type{ 1.8849404372 }, plssvm::real_type{ 1.0051856432 }, plssvm::real_type{ 0.29849993305 }, plssvm::real_type{ 1.6464627049 } },
                                                                   { plssvm::real_type{ -0.20981208921 }, plssvm::real_type{ 0.60276937379 }, plssvm::real_type{ -0.13086851759 }, plssvm::real_type{ 0.10805254527 } },
                                                                   { plssvm::real_type{ -1.1256816276 }, plssvm::real_type{ 2.1254153434 }, plssvm::real_type{ -0.16512657655 }, plssvm::real_type{ 2.5164553141 } } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(model.support_vectors(), support_vectors);
}

TYPED_TEST(RegressionModel, Labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // check labels getter
    ASSERT_FALSE(model.labels().has_value());
}

TYPED_TEST(RegressionModel, Weights) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct weights
    const std::vector<std::vector<plssvm::real_type>> correct_weights{
        { plssvm::real_type{ -1.8568721894e-01 }, plssvm::real_type{ 9.0116552290e-01 }, plssvm::real_type{ -2.2483112395e-01 }, plssvm::real_type{ 1.4909749921e-02 }, plssvm::real_type{ -4.5666857706e-01 }, plssvm::real_type{ -4.8888352876e-02 } }
    };
    EXPECT_FLOATING_POINT_MATRIX_EQ(model.weights().front(), (plssvm::aos_matrix<plssvm::real_type>{ correct_weights }));
}

TYPED_TEST(RegressionModel, Rho) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // test for the correct rho (bias) value
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.32260160011873423 } };
    EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), rho);
}

TYPED_TEST(RegressionModel, NumIters) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::regression_model<label_type> model{ this->filename };

    // check different_labels getter
    EXPECT_FALSE(model.num_iters().has_value());
}

template <typename T>
class RegressionModelSave : public ::testing::Test,
                            private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(RegressionModelSave, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(RegressionModelSave, Save) {
    using label_type = typename TestFixture::fixture_label_type;

    for (const plssvm::kernel_function_type kernel_function : util::kernel_functions_to_test) {
        const util::temporary_file model_file;
        util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/regression/6x4_TEMPLATE.libsvm.model", model_file.filename, kernel_function);

        // create a model using an existing LIBSVM model file
        const plssvm::regression_model<label_type> model{ model_file.filename };

        // write model to file
        model.save(model_file.filename);

        // read previously written file
        plssvm::detail::io::file_reader reader{ model_file.filename };
        reader.read_lines('#');
        // copy read lines
        std::vector<std::string_view> lines{ reader.lines() };

        // create vector containing correct regex expressions for the LIBSVM model file header
        std::vector<std::string> regex_patterns;
        regex_patterns.emplace_back("svm_type c_svr");
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
            case plssvm::kernel_function_type::laplacian:
            case plssvm::kernel_function_type::chi_squared:
                regex_patterns.emplace_back("gamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
                break;
            case plssvm::kernel_function_type::sigmoid:
                regex_patterns.emplace_back("gamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
                regex_patterns.emplace_back("coef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
                break;
        }
        regex_patterns.emplace_back("nr_class 2");
        regex_patterns.emplace_back("total_sv [0-9]+");
        regex_patterns.emplace_back("rho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
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
                FAIL() << fmt::format(R"(Can't find a line matching the regex pattern: "{}".)", pattern);
            }
        }

        // only support vectors should be left -> check the remaining lines if they match the correct pattern
        const std::string support_vector_pattern{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}" };
        for (const std::string_view line : lines) {
            const std::regex reg(support_vector_pattern, std::regex::extended);
            EXPECT_TRUE(std::regex_match(std::string{ line }, reg)) << fmt::format(R"(Line "{}" doesn't match the regex pattern "{}".)", line, support_vector_pattern);
        }
    }
}
