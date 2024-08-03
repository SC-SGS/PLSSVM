/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different gamma types.
 */

#include "plssvm/gamma.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::generate_specific_matrix

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, ASSERT_TRUE, ASSERT_FALSE

#include <sstream>  // std::istringstream
#include <string>   // std::string
#include <tuple>    // std::ignore
#include <variant>  // std::variant, std::holds_alternative, std::get

// check whether the plssvm::gamma_coefficient_type -> std::string conversions are correct
TEST(GammaCoefficientType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::gamma_coefficient_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::gamma_coefficient_type::scale, "scale");
}

TEST(GammaCoefficientType, to_string_unknown) {
    // check conversions to std::string from unknown gamma_coefficient_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::gamma_coefficient_type>(2), "unknown");
}

// check whether the std::string -> plssvm::gamma_coefficient_type conversions are correct
TEST(GammaCoefficientType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::gamma_coefficient_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOmatic", plssvm::gamma_coefficient_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("auto", plssvm::gamma_coefficient_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("scale", plssvm::gamma_coefficient_type::scale);
    EXPECT_CONVERSION_FROM_STRING("SCALE", plssvm::gamma_coefficient_type::scale);
}

TEST(GammaCoefficientType, from_string_unknown) {
    // foo isn't a valid gamma_coefficient_type
    std::istringstream input{ "foo" };
    plssvm::gamma_coefficient_type gamma_coef{};
    input >> gamma_coef;
    EXPECT_TRUE(input.fail());
}

// check whether the plssvm::gamma_type -> std::string conversions are correct
TEST(GammaType, to_string) {
    // check conversions to std::string
    plssvm::gamma_type gamma_value = plssvm::real_type{ 1.5 };
    EXPECT_CONVERSION_TO_STRING(gamma_value, "1.5");
    gamma_value = plssvm::gamma_coefficient_type::automatic;
    EXPECT_CONVERSION_TO_STRING(gamma_value, "automatic");
    gamma_value = plssvm::gamma_coefficient_type::scale;
    EXPECT_CONVERSION_TO_STRING(gamma_value, "scale");
}

TEST(GammaType, to_string_unknown) {
    // check conversions to std::string from unknown gamma_type
    const plssvm::gamma_type gamma_value = static_cast<plssvm::gamma_coefficient_type>(2);
    EXPECT_CONVERSION_TO_STRING(gamma_value, "unknown");
}

// check whether the std::string -> plssvm::gamma_type conversions are correct
TEST(GammaType, from_string) {
    // check conversion from std::string
    plssvm::gamma_type gamma_value = plssvm::real_type{ 1.5 };
    EXPECT_CONVERSION_FROM_STRING("1.5", gamma_value);
    gamma_value = plssvm::gamma_coefficient_type::automatic;
    EXPECT_CONVERSION_FROM_STRING("AUTOmatic", gamma_value);
    EXPECT_CONVERSION_FROM_STRING("auto", gamma_value);
    gamma_value = plssvm::gamma_coefficient_type::scale;
    EXPECT_CONVERSION_FROM_STRING("scale", gamma_value);
}

TEST(GammaType, from_string_unknown) {
    // foo isn't a valid gamma_type
    std::istringstream input{ "foo" };
    plssvm::gamma_type gamma_value{};
    input >> gamma_value;
    EXPECT_TRUE(input.fail());
}

TEST(GammaType, calculate_gamma_value_real_type) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::real_type{ 1.5 };

    // create a dummy matrix representing the actual data
    const auto matr = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 8, 4 });

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(gamma_value));
    // check the variant value
    EXPECT_FLOATING_POINT_EQ(plssvm::calculate_gamma_value(gamma_value, matr), plssvm::real_type{ 1.5 });
}

TEST(GammaType, calculate_gamma_value_gamma_coefficient_type_automatic) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::gamma_coefficient_type::automatic;

    // create a dummy matrix representing the actual data
    const auto matr = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 8, 4 });

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::gamma_coefficient_type>(gamma_value));
    // check the variant value -> 1 / num_features
    EXPECT_FLOATING_POINT_EQ(plssvm::calculate_gamma_value(gamma_value, matr), plssvm::real_type{ 0.25 });
}

TEST(GammaType, calculate_gamma_value_gamma_coefficient_type_scale) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::gamma_coefficient_type::scale;

    // create a dummy matrix representing the actual data
    const auto matr = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 8, 4 });

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::gamma_coefficient_type>(gamma_value));
    // check the variant value -> 1 / (num_features * variance(matr))
    EXPECT_FLOATING_POINT_NEAR(plssvm::calculate_gamma_value(gamma_value, matr), plssvm::real_type{ 0.047505938242280283668 });
}

TEST(GammaType, get_gamma_string_real_type) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::real_type{ 1.5 };

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(gamma_value));
    // check the variant string
    EXPECT_EQ(plssvm::get_gamma_string(gamma_value), fmt::format("{}", plssvm::real_type{ 1.5 }));
}

TEST(GammaType, get_gamma_string_gamma_coefficient_type_automatic) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::gamma_coefficient_type::automatic;

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::gamma_coefficient_type>(gamma_value));
    // check the variant string
    EXPECT_EQ(plssvm::get_gamma_string(gamma_value), std::string{ "\"1 / num_features\"" });
}

TEST(GammaType, get_gamma_string_gamma_coefficient_type_scale) {
    // create a gamma_type with a real_type value
    const plssvm::gamma_type gamma_value = plssvm::gamma_coefficient_type::scale;

    // the std::variant must hold the real_type member
    ASSERT_TRUE(std::holds_alternative<plssvm::gamma_coefficient_type>(gamma_value));
    // check the variant string
    EXPECT_EQ(plssvm::get_gamma_string(gamma_value), std::string{ "\"1 / (num_features * variance(input_data))\"" });
}
