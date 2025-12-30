/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the custom min-max scaler.
 */

#include "plssvm/data_set/min_max_scaler.hpp"  // plssvm::min_max_scaler

#include "plssvm/constants.hpp"                 // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::min_max_scaler_exception
#include "plssvm/matrix.hpp"                    // plssvm::soa_matrix

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ, EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::{temporary_file, redirect_output}

#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, ASSERT_EQ, ASSERT_GE, ASSERT_TRUE, ::testing::Test

#include <cstddef>  // std::size_t
#include <regex>    // std::regex, std::regex::extended, std::regex_match
#include <vector>   // std::vector

TEST(MinMaxScaler, DefaultConstructFactor) {
    // create factor
    const plssvm::min_max_scaler::factors factor{};

    // test values
    EXPECT_EQ(factor.feature, std::size_t{});
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{});
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{});
}

TEST(MinMaxScaler, ConstructFactor) {
    // create factor
    const plssvm::min_max_scaler::factors factor{ 1, plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } };

    // test values
    EXPECT_EQ(factor.feature, 1);
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{ -2.5 });
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{ 2.5 });
}

TEST(MinMaxScaler, ConstructInterval) {
    // create scaling class
    const plssvm::min_max_scaler scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // test whether the values have been correctly set
    EXPECT_FLOATING_POINT_EQ(scaler.scaling_interval().first, plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(scaler.scaling_interval().second, plssvm::real_type{ 1.0 });
    EXPECT_FALSE(scaler.scaling_factors().has_value());
}

TEST(MinMaxScaler, ConstructInvalidInterval) {
    // create scaling class with an invalid interval
    EXPECT_THROW_WHAT((plssvm::min_max_scaler{ plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } }),
                      plssvm::min_max_scaler_exception,
                      "Inconsistent scaling interval specification: lower (1) must be less than upper (-1)!");
}

TEST(MinMaxScaler, ConstructFromFile) {
    using factors_type = plssvm::min_max_scaler::factors;

    // create scaling class
    const plssvm::min_max_scaler scaler{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // test whether the values have been correctly set
    EXPECT_EQ(scaler.scaling_interval().first, plssvm::detail::convert_to<plssvm::real_type>("-1.4"));
    EXPECT_EQ(scaler.scaling_interval().second, plssvm::detail::convert_to<plssvm::real_type>("2.6"));
    const std::vector<factors_type> correct_factors = {
        factors_type{ 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } },
        factors_type{ 1, plssvm::real_type{ 1.1 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.3 }, plssvm::real_type{ 4.3 } },
        factors_type{ 4, plssvm::real_type{ 4.4 }, plssvm::real_type{ 5.4 } },
    };
    const auto &factors_opt = scaler.scaling_factors();
    ASSERT_TRUE(factors_opt.has_value());
    if (factors_opt.has_value()) {
        const std::vector<plssvm::min_max_scaler::factors> factors = factors_opt.value();
        ASSERT_EQ(factors.size(), correct_factors.size());
        for (std::size_t i = 0; i < factors.size(); ++i) {
            EXPECT_EQ(factors[i].feature, correct_factors[i].feature);
            EXPECT_FLOATING_POINT_EQ(factors[i].lower, correct_factors[i].lower);
            EXPECT_FLOATING_POINT_EQ(factors[i].upper, correct_factors[i].upper);
        }
    }
}

TEST(MinMaxScaler, Save) {
    // create scaling class
    const plssvm::min_max_scaler scaler{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scaler.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check file content
    ASSERT_GE(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
    reg = std::regex{ "\\+?[1-9]+[0-9]* [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    for (std::size_t i = 2; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}

TEST(MinMaxScaler, SaveEmptyScalingFactors) {
    // create scaling class
    const plssvm::min_max_scaler scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scaler.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check the content
    ASSERT_EQ(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    const std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
}

TEST(MinMaxScaler, ScaleScalingFactorsEmpty) {
    // create scaling class
    plssvm::min_max_scaler scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // create data and ground truth result
    auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 10, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto [data_scaled, scaling_factors] = util::scale(data, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });

    // scale the data (inplace)
    scaler.scale(data);

    // check whether scaling was successful
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, data_scaled);
}

TEST(MinMaxScaler, ScaleScalingFactors) {
    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope

    // create data and ground truth result
    auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 10, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    auto data_2 = data;
    const auto [data_scaled, scaling_factors] = util::scale(data, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 });

    {
        // create scaling class
        plssvm::min_max_scaler scaler{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };
        // scale
        scaler.scale(data);
        // save scaling factors
        scaler.save(tmp_file.filename);
    }

    // create scaling class
    plssvm::min_max_scaler scaler{ tmp_file.filename };

    // scale the data (inplace)
    scaler.scale(data_2);

    // check whether scaling was successful
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, data_scaled);
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data_2, data_scaled);
}

TEST(MinMaxScaler, ScaleTooManyScalingFactors) {
    // create scaling class
    plssvm::min_max_scaler scaler{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create data and ground truth result
    auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 10, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // invalid number of scaling factors
    EXPECT_THROW_WHAT_MATCHER(scaler.scale(data), plssvm::min_max_scaler_exception, ::testing::HasSubstr("Need at most as much scaling factors as features in the data set are present (3), but 4 were given!"));
}

TEST(MinMaxScaler, ScaleFeatureIndexTooBig) {
    // create scaling class
    plssvm::min_max_scaler scaler{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create data and ground truth result
    auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 10, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // invalid number of scaling factors
    EXPECT_THROW_WHAT(scaler.scale(data), plssvm::min_max_scaler_exception, "The maximum scaling feature index most not be greater or equal than 4, but is 4!");
}

TEST(MinMaxScaler, ScaleScalingFactorMoreThanOnce) {
    // create scaling class
    plssvm::min_max_scaler scaler{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/feature_index_more_than_once.txt" };

    // create data and ground truth result
    auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 10, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // invalid number of scaling factors
    EXPECT_THROW_WHAT(scaler.scale(data), plssvm::min_max_scaler_exception, "Found more than one scaling factor for the feature index 0!");
}
