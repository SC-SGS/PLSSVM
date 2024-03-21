/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different kernel_types.
 */

#include "plssvm/kernel_function_types.hpp"

#include "tests/custom_test_macros.hpp"     // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE

#include <sstream>  // std::istringstream

// check whether the plssvm::kernel_function_type -> std::string conversions are correct
TEST(KernelType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::linear, "linear");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::polynomial, "polynomial");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::rbf, "rbf");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::sigmoid, "sigmoid");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::laplacian, "laplacian");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::chi_squared, "chi_squared");
}

TEST(KernelType, to_string_unknown) {
    // check conversions to std::string from unknown kernel_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::kernel_function_type>(6), "unknown");
}

// check whether the std::string -> plssvm::kernel_function_type conversions are correct
TEST(KernelType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("linear", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("LINEAR", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("0", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("polynomial", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("POLynomIAL", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("poly", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("1", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("rbf", plssvm::kernel_function_type::rbf);
    EXPECT_CONVERSION_FROM_STRING("rBf", plssvm::kernel_function_type::rbf);
    EXPECT_CONVERSION_FROM_STRING("2", plssvm::kernel_function_type::rbf);
    EXPECT_CONVERSION_FROM_STRING("sigmoid", plssvm::kernel_function_type::sigmoid);
    EXPECT_CONVERSION_FROM_STRING("SIGMOID", plssvm::kernel_function_type::sigmoid);
    EXPECT_CONVERSION_FROM_STRING("3", plssvm::kernel_function_type::sigmoid);
    EXPECT_CONVERSION_FROM_STRING("laplacian", plssvm::kernel_function_type::laplacian);
    EXPECT_CONVERSION_FROM_STRING("Laplacian", plssvm::kernel_function_type::laplacian);
    EXPECT_CONVERSION_FROM_STRING("4", plssvm::kernel_function_type::laplacian);
    EXPECT_CONVERSION_FROM_STRING("chi_squared", plssvm::kernel_function_type::chi_squared);
    EXPECT_CONVERSION_FROM_STRING("CHI-squared", plssvm::kernel_function_type::chi_squared);
    EXPECT_CONVERSION_FROM_STRING("5", plssvm::kernel_function_type::chi_squared);
}

TEST(KernelType, from_string_unknown) {
    // foo isn't a valid kernel_type
    std::istringstream input{ "foo" };
    plssvm::kernel_function_type kernel{};
    input >> kernel;
    EXPECT_TRUE(input.fail());
}

// check whether the plssvm::kernel_function_type -> math string conversions are correct
TEST(KernelType, kernel_to_math_string) {
    // check conversion from plssvm::kernel_function_type to the respective math function string
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::linear), "u'*v");
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::polynomial), "(gamma*u'*v+coef0)^degree");
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::rbf), "exp(-gamma*|u-v|^2)");
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::sigmoid), "tanh(gamma*u'*v+coef0)");
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::laplacian), "exp(-gamma*|u-v|_1)");
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(plssvm::kernel_function_type::chi_squared), "exp(-gamma*sum_i((x[i]-y[i])^2/(x[i]+y[i])))");
}

TEST(KernelType, kernel_to_math_string_unkown) {
    // check conversion from an unknown plssvm::kernel_function_type to the (non-existing) math string
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(static_cast<plssvm::kernel_function_type>(6)), "unknown");
}
