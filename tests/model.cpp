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

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::{convert_to, split_as}
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "utility.hpp"  // util::create_temp_file

#include "gtest/gtest.h"  // EXPECT_EQ, EXPECT_TRUE, ASSERT_GT, GTEST_FAIL, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                          // ::testing::{Types, StaticAssertTypeEq, Test, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <iostream>     // std::cout
#include <regex>        // std::regex, std::regex_match, std::regex::extended
#include <sstream>      // std::stringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string
#include <string_view>  // std::string_view
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

template <typename T>
class Model : public ::testing::Test {
    void SetUp() override {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }
    void TearDown() override {
        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};
TYPED_TEST_SUITE(Model, type_combinations_types);

TYPED_TEST(Model, construct) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };

    // test for correct construction
    EXPECT_EQ(model.num_support_vectors(), 5);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.svm_parameter(), plssvm::parameter<real_type>{});
    const std::vector<std::vector<real_type>> support_vectors{
        plssvm::detail::split_as<real_type>("-1.117828e+00 -2.908719e+00 6.663834e-01 1.097883e+00"),
        plssvm::detail::split_as<real_type>("-5.282118e-01 -3.358810e-01 5.168730e-01 5.460446e-01"),
        plssvm::detail::split_as<real_type>("-2.098121e-01 6.027694e-01 -1.308685e-01 1.080525e-01"),
        plssvm::detail::split_as<real_type>("1.884940e+00 1.005186e+00 2.984999e-01 1.646463e+00"),
        plssvm::detail::split_as<real_type>("5.765022e-01 1.014056e+00 1.300943e-01 7.261914e-01")
    };
    EXPECT_EQ(model.support_vectors(), support_vectors);
    EXPECT_EQ(model.weights(), plssvm::detail::split_as<real_type>("-0.17609610490769723 0.8838187731213127 -0.47971257671001616 0.0034556484621847128 -0.23146573996578407"));
    EXPECT_EQ(model.rho(), plssvm::detail::convert_to<real_type>("0.37330625882191915"));
}

TYPED_TEST(Model, typedefs) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };

    // test internal typedefs
    ::testing::StaticAssertTypeEq<real_type, typename decltype(model)::real_type>();
    ::testing::StaticAssertTypeEq<label_type, typename decltype(model)::label_type>();
}

TYPED_TEST(Model, num_support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct number of support vectors
    EXPECT_EQ(model.num_support_vectors(), 5);
}
TYPED_TEST(Model, num_features) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct number of features
    EXPECT_EQ(model.num_features(), 4);
}
TYPED_TEST(Model, svm_parameter) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct number of features
    EXPECT_EQ(model.svm_parameter(), plssvm::parameter<real_type>{});
}
TYPED_TEST(Model, support_vectors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct support vectors
    const std::vector<std::vector<real_type>> support_vectors{
        plssvm::detail::split_as<real_type>("-1.117828e+00 -2.908719e+00 6.663834e-01 1.097883e+00"),
        plssvm::detail::split_as<real_type>("-5.282118e-01 -3.358810e-01 5.168730e-01 5.460446e-01"),
        plssvm::detail::split_as<real_type>("-2.098121e-01 6.027694e-01 -1.308685e-01 1.080525e-01"),
        plssvm::detail::split_as<real_type>("1.884940e+00 1.005186e+00 2.984999e-01 1.646463e+00"),
        plssvm::detail::split_as<real_type>("5.765022e-01 1.014056e+00 1.300943e-01 7.261914e-01")
    };
    EXPECT_EQ(model.support_vectors(), support_vectors);
}
TYPED_TEST(Model, weights) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct weights
    EXPECT_EQ(model.weights(), plssvm::detail::split_as<real_type>("-0.17609610490769723 0.8838187731213127 -0.47971257671001616 0.0034556484621847128 -0.23146573996578407"));
}
TYPED_TEST(Model, rho) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // create a model using an existing LIBSVM model file
    plssvm::model<real_type, label_type> model{ PLSSVM_TEST_PATH "/data/model/5x4_linear.libsvm.model" };
    // test for the correct rho (bias) value
    EXPECT_EQ(model.rho(), plssvm::detail::convert_to<real_type>("0.37330625882191915"));
}

class ModelSave : public ::testing::TestWithParam<std::string> {
    void SetUp() override {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }
    void TearDown() override {
        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};
TEST_P(ModelSave, save) {
    // create a model using an existing LIBSVM model file
    plssvm::model<double, int> model{ fmt::format("{}{}", PLSSVM_TEST_PATH, GetParam()) };

    // create temporary file
    const std::string filename = util::create_temp_file();

    {
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
        regex_patterns.emplace_back(fmt::format("kernel_type {}", model.svm_parameter().kernel));
        switch (model.svm_parameter().kernel) {
            case plssvm::kernel_type::linear:
                break;
            case plssvm::kernel_type::polynomial:
                regex_patterns.emplace_back("degree [0-9]+");
                regex_patterns.emplace_back("gamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
                regex_patterns.emplace_back("coef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
                break;
            case plssvm::kernel_type::rbf:
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
                GTEST_FAIL() << fmt::format("Can't find a line matching the regex pattern: \"{}\"", pattern);
            }
        }
        // only support vectors should be left -> check the remaining lines if they match the correct pattern
        const std::string support_vector_pattern{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?)*" };
        for (const std::string_view line : lines) {
            std::regex reg(support_vector_pattern, std::regex::extended);
            EXPECT_TRUE(std::regex_match(std::string{ line }, reg)) << fmt::format("Line \"{}\" doesn't match the regex pattern \"{}\"", line, support_vector_pattern);
        }
    }

    // remove temporary file
    std::filesystem::remove(filename);
}
INSTANTIATE_TEST_SUITE_P(Model, ModelSave, ::testing::Values("/data/model/5x4_linear.libsvm.model", "/data/model/5x4_polynomial.libsvm.model", "/data/model/5x4_rbf.libsvm.model"));
