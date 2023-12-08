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

#include "plssvm/classification_types.hpp"   // plssvm::classification_type, plssvm::calculate_number_of_classifiers
#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_EQ
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_classification_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_num_classes, get_distinct_label, get_correct_model_file_labels}

#include "gtest/gtest-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, ASSERT_GT, FAIL,
                                   // ::testing::{Test, StaticAssertTypeEq}

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <regex>        // std::regex, std::regex_match, std::regex::extended
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

template <typename T>
class Model : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        const std::string template_filename = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_{}_TEMPLATE.libsvm.model", util::get_num_classes<fixture_label_type>(), fixture_classification);
        util::instantiate_template_file<fixture_label_type>(template_filename, this->filename);
    }
};
TYPED_TEST_SUITE(Model, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(Model, typedefs) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<label_type, typename decltype(model)::label_type>();
    ::testing::StaticAssertTypeEq<std::size_t, typename decltype(model)::size_type>();
}

TYPED_TEST(Model, construct) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create model
    const plssvm::model<label_type> model{ this->filename };
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();

    // test for correct construction
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params(), plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::linear });
    EXPECT_EQ(model.support_vectors().shape(), (std::array<std::size_t, 2>{ 6, 4 }));
    EXPECT_EQ(model.labels().size(), 6);
    EXPECT_EQ(model.num_classes(), num_classes_for_label_type);
    EXPECT_EQ(model.classes(), util::get_distinct_label<label_type>());
    if constexpr (classification == plssvm::classification_type::oaa) {
        // OAA
        EXPECT_EQ(model.weights().size(), 1);
        EXPECT_EQ(model.weights().front().shape(), (std::array<std::size_t, 2>{ num_classes_for_label_type, 6 }));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // OAO
        EXPECT_EQ(model.weights().size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
    } else {
        FAIL() << "unknown classification type";
    }
    EXPECT_EQ(model.rho().size(), plssvm::calculate_number_of_classifiers(classification, model.num_classes()));
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_FALSE(model.num_iters().has_value());
}

TYPED_TEST(Model, num_support_vectors) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // test for the correct number of support vectors
    EXPECT_EQ(model.num_support_vectors(), 6);
}
TYPED_TEST(Model, num_features) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // test for the correct number of features
    EXPECT_EQ(model.num_features(), 4);
}
TYPED_TEST(Model, get_params) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // test for the correct number of features
    EXPECT_EQ(model.get_params(), plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::linear });
}
TYPED_TEST(Model, support_vectors) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // test for the correct support vectors
    const plssvm::soa_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ -1.1178275006 }, plssvm::real_type{ -2.9087188881 }, plssvm::real_type{ 0.66638344270 }, plssvm::real_type{ 1.0978832704 } },
                                                                   { plssvm::real_type{ -0.52821182989 }, plssvm::real_type{ -0.33588098497 }, plssvm::real_type{ 0.51687296030 }, plssvm::real_type{ 0.54604461446 } },
                                                                   { plssvm::real_type{ 0.57650218263 }, plssvm::real_type{ 1.0140559662 }, plssvm::real_type{ 0.13009428080 }, plssvm::real_type{ 0.72619138869 } },
                                                                   { plssvm::real_type{ 1.8849404372 }, plssvm::real_type{ 1.0051856432 }, plssvm::real_type{ 0.29849993305 }, plssvm::real_type{ 1.6464627049 } },
                                                                   { plssvm::real_type{ -0.20981208921 }, plssvm::real_type{ 0.60276937379 }, plssvm::real_type{ -0.13086851759 }, plssvm::real_type{ 0.10805254527 } },
                                                                   { plssvm::real_type{ -1.1256816276 }, plssvm::real_type{ 2.1254153434 }, plssvm::real_type{ -0.16512657655 }, plssvm::real_type{ 2.5164553141 } } },
                                                                 plssvm::PADDING_SIZE,
                                                                 plssvm::PADDING_SIZE };
    EXPECT_FLOATING_POINT_MATRIX_EQ(model.support_vectors(), support_vectors);
}
TYPED_TEST(Model, labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // check labels getter
    EXPECT_EQ(model.labels(), util::get_correct_model_file_labels<label_type>());
}
TYPED_TEST(Model, num_classes) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // check num_different_labels getter
    EXPECT_EQ(model.num_classes(), util::get_num_classes<label_type>());
}
TYPED_TEST(Model, classes) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // check different_labels getter
    EXPECT_EQ(model.classes(), util::get_distinct_label<label_type>());
}
TYPED_TEST(Model, weights) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create model
    const plssvm::model<label_type> model{ this->filename };
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();

    // test for the correct weights
    std::vector<std::vector<plssvm::real_type>> correct_weights{
        { plssvm::real_type{ -1.8568721894e-01 }, plssvm::real_type{ 9.0116552290e-01 }, plssvm::real_type{ -2.2483112395e-01 }, plssvm::real_type{ 1.4909749921e-02 }, plssvm::real_type{ -4.5666857706e-01 }, plssvm::real_type{ -4.8888352876e-02 } },
        { plssvm::real_type{ 1.1365048527e-01 }, plssvm::real_type{ -3.2357185930e-01 }, plssvm::real_type{ 8.9871548758e-01 }, plssvm::real_type{ -7.5259922896e-02 }, plssvm::real_type{ -4.7955922738e-01 }, plssvm::real_type{ -1.3397496327e-01 } },
        { plssvm::real_type{ 2.8929914669e-02 }, plssvm::real_type{ -4.8559849173e-01 }, plssvm::real_type{ -5.6740083618e-01 }, plssvm::real_type{ 8.7841608802e-02 }, plssvm::real_type{ 9.7960957282e-01 }, plssvm::real_type{ -4.3381768383e-02 } },
        { plssvm::real_type{ 4.3106819001e-02 }, plssvm::real_type{ -9.1995171877e-02 }, plssvm::real_type{ -1.0648352745e-01 }, plssvm::real_type{ -2.7491435827e-02 }, plssvm::real_type{ -4.3381768383e-02 }, plssvm::real_type{ 2.2624508453e-01 } }
    };

    // check for correct weights
    if constexpr (classification == plssvm::classification_type::oaa) {
        // OAA
        ASSERT_EQ(model.weights().size(), 1);
        ASSERT_EQ(model.weights().front().shape(), (std::array<std::size_t, 2>{ num_classes_for_label_type, 6 }));

        switch (num_classes_for_label_type) {
            case 2:
                // ignore last two weight vectors
                correct_weights.pop_back();
                correct_weights.pop_back();
                break;
            case 3:
                // ignore last weight vector
                correct_weights.pop_back();
                break;
            case 4:
                // use full weight vector
                break;
            default:
                FAIL() << "Unreachable!";
                break;
        }
        EXPECT_FLOATING_POINT_MATRIX_EQ(model.weights().front(), (plssvm::aos_matrix<plssvm::real_type>{ correct_weights, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // OAO
        ASSERT_EQ(model.weights().size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);

        std::vector<plssvm::aos_matrix<plssvm::real_type>> weights(num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
        switch (num_classes_for_label_type) {
            case 2:
                // 0vs1
                weights.front() = plssvm::aos_matrix<plssvm::real_type>{ { correct_weights.front() } };
                break;
            case 3:
                // 0vs1
                weights[0] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[0][0], correct_weights[0][1], correct_weights[0][2], correct_weights[0][3] } } };
                // 0vs2
                weights[1] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[1][0], correct_weights[1][1], correct_weights[0][4], correct_weights[0][5] } } };
                // 1vs2
                weights[2] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[1][2], correct_weights[1][3], correct_weights[1][4], correct_weights[1][5] } } };
                break;
            case 4:
                // 0vs1
                weights[0] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[0][0], correct_weights[0][1], correct_weights[0][2], correct_weights[0][3] } } };
                // 0vs2
                weights[1] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[1][0], correct_weights[1][1], correct_weights[0][4] } } };
                // 0vs3
                weights[2] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[2][0], correct_weights[2][1], correct_weights[0][5] } } };
                // 1vs2
                weights[3] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[1][2], correct_weights[1][3], correct_weights[1][4] } } };
                // 1vs3
                weights[4] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[2][2], correct_weights[2][3], correct_weights[1][5] } } };
                // 2vs3
                weights[5] = plssvm::aos_matrix<plssvm::real_type>{ { { correct_weights[2][4], correct_weights[2][5] } } };
                break;
            default:
                FAIL() << "Unreachable!";
                break;
        }
        ASSERT_EQ(model.weights().size(), weights.size());
        for (std::size_t i = 0; i < weights.size(); ++i) {
            EXPECT_FLOATING_POINT_MATRIX_EQ(model.weights()[i], (plssvm::aos_matrix<plssvm::real_type>{ weights[i], plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
        }
    } else {
        FAIL() << "unknown classification type";
    }
}
TYPED_TEST(Model, rho) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create model
    const plssvm::model<label_type> model{ this->filename };
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();

    // test for the correct rho (bias) value
    const std::vector<plssvm::real_type> all_rhos{
        plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 }, plssvm::real_type{ -0.2415331131484474 }, plssvm::real_type{ -2.636779683484747e-16 }
    };

    if constexpr (classification == plssvm::classification_type::oaa) {
        // one vs. all classification
        ASSERT_EQ(model.rho().size(), num_classes_for_label_type);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2], all_rhos[3] }));
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // one vs. all classification
        ASSERT_EQ(model.rho().size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<plssvm::real_type>{ all_rhos[0] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(model.rho(), all_rhos);
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else {
        FAIL() << "unknown classification type";
    }
}
TYPED_TEST(Model, get_classification_type) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // check different_labels getter
    EXPECT_EQ(model.get_classification_type(), classification);
}
TYPED_TEST(Model, num_iters) {
    using label_type = typename TestFixture::fixture_label_type;

    // create model
    const plssvm::model<label_type> model{ this->filename };

    // check different_labels getter
    EXPECT_FALSE(model.num_iters().has_value());
}

template <typename T>
class ModelSave : public ::testing::Test, private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;
};
TYPED_TEST_SUITE(ModelSave, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ModelSave, save) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    for (const plssvm::kernel_function_type kernel_function : { plssvm::kernel_function_type::linear, plssvm::kernel_function_type::polynomial, plssvm::kernel_function_type::rbf }) {
        const std::size_t num_classes = util::get_num_classes<label_type>();

        const util::temporary_file model_file;
        const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_{}_{}_TEMPLATE.libsvm.model", util::get_num_classes<label_type>(), kernel_function, classification);
        util::instantiate_template_file<label_type>(template_file_name, model_file.filename);

        // create a model using an existing LIBSVM model file
        const plssvm::model<label_type> model{ model_file.filename };

        // write model to file
        model.save(model_file.filename);

        // read previously written file
        plssvm::detail::io::file_reader reader{ model_file.filename };
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
        regex_patterns.emplace_back(fmt::format("label (.+ ?){{{}}}", num_classes));
        regex_patterns.emplace_back("total_sv [0-9]+");
        regex_patterns.emplace_back(fmt::format("nr_sv ([0-9]+ ?){{{}}}", num_classes));
        regex_patterns.emplace_back(fmt::format("rho ([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){{{}}}", plssvm::calculate_number_of_classifiers(classification, num_classes)));
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
        const std::string support_vector_pattern{ fmt::format("([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){{{}}} ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){{4}}",
                                                              classification == plssvm::classification_type::oaa ? num_classes : num_classes - 1) };
        for (const std::string_view line : lines) {
            const std::regex reg(support_vector_pattern, std::regex::extended);
            EXPECT_TRUE(std::regex_match(std::string{ line }, reg)) << fmt::format(R"(Line "{}" doesn't match the regex pattern "{}".)", line, support_vector_pattern);
        }
    }
}