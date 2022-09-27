/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the data_set used for learning an SVM model.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::{convert_to, split_as}
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "utility.hpp"  // util::create_temp_file, util::redirect_output

#include "gtest/gtest.h"  // EXPECT_EQ, EXPECT_TRUE, ASSERT_GT, GTEST_FAIL, TYPED_TEST, TYPED_TEST_SUITE, TEST_P, INSTANTIATE_TEST_SUITE_P
                          // ::testing::{Types, StaticAssertTypeEq, Test, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <regex>        // std::regex, std::regex_match, std::regex::extended
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

////////////////////////////////////////////////////////////////////////////////
////                          scaling nested-class                          ////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DataSetScaling : public ::testing::Test, private util::redirect_output {};
TYPED_TEST_SUITE(DataSetScaling, type_combinations_types);

TYPED_TEST(DataSetScaling, default_construct_factor) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    factor_type factor{};

    // test values
    EXPECT_EQ(factor.feature, std::size_t{});
    EXPECT_EQ(factor.lower, real_type{});
    EXPECT_EQ(factor.upper, real_type{});
}
TYPED_TEST(DataSetScaling, construct_factor) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    factor_type factor{ 1, real_type{ -2.5 }, real_type{ 2.5 } };

    // test values
    EXPECT_EQ(factor.feature, 1);
    EXPECT_EQ(factor.lower, -2.5);
    EXPECT_EQ(factor.upper, 2.5);
}

TYPED_TEST(DataSetScaling, construct_interval) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ real_type{ -1.0 }, real_type{ 1.0 } };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, real_type{ -1.0 });
    EXPECT_EQ(scale.scaling_interval.second, real_type{ 1.0 });
    EXPECT_TRUE(scale.scaling_factors.empty());
}
TYPED_TEST(DataSetScaling, construct_from_file) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, plssvm::detail::convert_to<real_type>("-1.4"));
    EXPECT_EQ(scale.scaling_interval.second, plssvm::detail::convert_to<real_type>("2.6"));
    std::vector<factors_type> correct_factors = {
        factors_type{ 0, real_type{ 0.0 }, real_type{ 1.0 } },
        factors_type{ 1, real_type{ 1.1 }, real_type{ 2.1 } },
        factors_type{ 3, real_type{ 3.3 }, real_type{ 4.3 } },
        factors_type{ 4, real_type{ 4.4 }, real_type{ 5.4 } },
    };
    ASSERT_EQ(scale.scaling_factors.size(), correct_factors.size());
    for (std::size_t i = 0; i < correct_factors.size(); ++i) {
        EXPECT_EQ(scale.scaling_factors[i].feature, correct_factors[i].feature);
        EXPECT_EQ(scale.scaling_factors[i].lower, correct_factors[i].lower);
        EXPECT_EQ(scale.scaling_factors[i].upper, correct_factors[i].upper);
    }
}

TYPED_TEST(DataSetScaling, save) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create temporary file
    const std::string filename = util::create_temp_file();
    {
        // save scaling factors
        scale.save(filename);

        // read file and check its content
        plssvm::detail::io::file_reader reader{ filename };
        reader.read_lines('#');

        std::vector<std::string> regex_patterns;
        // header
        regex_patterns.emplace_back("x");
        regex_patterns.emplace_back("[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");
        // the remaining values
        regex_patterns.emplace_back("\\+?[1-9]+[0-9]* [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");

        // check the content
        ASSERT_GE(reader.num_lines(), 2);
        std::regex reg{ regex_patterns[0], std::regex::extended };  // check the line only containing an x
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(0) }, reg));
        reg = std::regex{ regex_patterns[1], std::regex::extended };  // check the scaling interval line
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
        reg = std::regex{ regex_patterns[2], std::regex::extended };  // check the remaining lines
        for (std::size_t i = 2; i < reader.num_lines(); ++i) {
            EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
        }
    }
    std::filesystem::remove(filename);
}
TYPED_TEST(DataSetScaling, save_empty_scaling_factors) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using scaling_type = typename plssvm::data_set<real_type, label_type>::scaling;

    // create scaling class
    scaling_type scale{ real_type{ -1.0 }, real_type{ 1.0 } };

    // create temporary file
    const std::string filename = util::create_temp_file();
    {
        // save scaling factors
        scale.save(filename);

        // read file and check its content
        plssvm::detail::io::file_reader reader{ filename };
        reader.read_lines('#');

        std::vector<std::string> regex_patterns;
        // header
        regex_patterns.emplace_back("x");
        regex_patterns.emplace_back("[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?");

        // check the content
        ASSERT_EQ(reader.num_lines(), 2);
        std::regex reg{ regex_patterns[0], std::regex::extended };  // check the line only containing an x
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(0) }, reg));
        reg = std::regex{ regex_patterns[1], std::regex::extended };  // check the scaling interval line
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
    }
    std::filesystem::remove(filename);
}

////////////////////////////////////////////////////////////////////////////////
////                       label mapper nested-class                        ////
////////////////////////////////////////////////////////////////////////////////

template <typename T>
class DataSetLabelMapper : public ::testing::Test {};
TYPED_TEST_SUITE(DataSetLabelMapper, type_combinations_types);

TYPED_TEST(DataSetLabelMapper, construct) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        different_labels = std::vector<int>{ -64, 32 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        different_labels = std::vector<std::string>{ "cat", "dog" };
    }

    // create label mapper
    const std::vector<label_type> labels = {
        different_labels[0],
        different_labels[1],
        different_labels[1],
        different_labels[0],
        different_labels[1]
    };
    const label_mapper_type mapper{ labels };

    // test values
    EXPECT_EQ(mapper.num_mappings(), 2);
    EXPECT_EQ(mapper.labels(), different_labels);
    // test mapping
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ -1 }), different_labels[0]);
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ 1 }), different_labels[1]);
    EXPECT_EQ(mapper.get_mapped_value_by_label(different_labels[0]), real_type{ -1 });
    EXPECT_EQ(mapper.get_mapped_value_by_label(different_labels[1]), real_type{ 1 });
}
TYPED_TEST(DataSetLabelMapper, construct_too_many_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    std::vector<label_type> different_labels;
    if constexpr (std::is_same_v<label_type, int>) {
        different_labels = std::vector<int>{ -64, 32, 128 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        different_labels = std::vector<std::string>{ "cat", "dog", "mouse" };
    }

    // too many labels provided!
    EXPECT_THROW_WHAT(label_mapper_type{ different_labels }, plssvm::data_set_exception, "Currently only binary classification is supported, but 3 different labels were given!");
}
TYPED_TEST(DataSetLabelMapper, get_mapped_value_by_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("-10")), real_type{ -1 });
    EXPECT_EQ(mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("10")), real_type{ 1 });
}
TYPED_TEST(DataSetLabelMapper, get_mapped_value_by_invalid_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_mapped_value_by_label(plssvm::detail::convert_to<label_type>("42")), plssvm::data_set_exception, "Label \"42\" unknown in this label mapping!");
}
TYPED_TEST(DataSetLabelMapper, get_label_by_mapped_value) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ -1 }), plssvm::detail::convert_to<label_type>("-10"));
    EXPECT_EQ(mapper.get_label_by_mapped_value(real_type{ 1 }), plssvm::detail::convert_to<label_type>("10"));
}
TYPED_TEST(DataSetLabelMapper, get_label_by_invalid_mapped_value) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-10"), plssvm::detail::convert_to<label_type>("10") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_label_by_mapped_value(real_type{ 0.0 }), plssvm::data_set_exception, "Mapped value \"0\" unknown in this label mapping!");
}
TYPED_TEST(DataSetLabelMapper, num_mappings) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-1"), plssvm::detail::convert_to<label_type>("1") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.num_mappings(), 2);
}
TYPED_TEST(DataSetLabelMapper, labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;
    using label_mapper_type = typename plssvm::data_set<real_type, label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = { plssvm::detail::convert_to<label_type>("-1"), plssvm::detail::convert_to<label_type>("1") };

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.labels(), different_labels);
}