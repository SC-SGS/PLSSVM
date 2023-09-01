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

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "backends/compare.hpp"       // compare::detail::{linear_kernel, poly_kernel, rbf_kernel}
#include "custom_test_macros.hpp"     // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_NEAR
#include "naming.hpp"                 // naming::plssvm::real_type_to_name
#include "utility.hpp"                // util::generate_random_vector

#include "gtest/gtest.h"              // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH

#include <array>                      // std::array
#include <cstddef>                    // std::size_t
#include <sstream>                    // std::istringstream
#include <tuple>                      // std::ignore
#include <vector>                     // std::vector

// check whether the plssvm::kernel_function_type -> std::string conversions are correct
TEST(KernelType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::linear, "linear");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::polynomial, "polynomial");
    EXPECT_CONVERSION_TO_STRING(plssvm::kernel_function_type::rbf, "rbf");
}
TEST(KernelType, to_string_unknown) {
    // check conversions to std::string from unknown kernel_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::kernel_function_type>(3), "unknown");
}

// check whether the std::string -> plssvm::kernel_function_type conversions are correct
TEST(KernelType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("linear", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("LINEAR", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("0", plssvm::kernel_function_type::linear);
    EXPECT_CONVERSION_FROM_STRING("polynomial", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("POLynomIAL", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("1", plssvm::kernel_function_type::polynomial);
    EXPECT_CONVERSION_FROM_STRING("rbf", plssvm::kernel_function_type::rbf);
    EXPECT_CONVERSION_FROM_STRING("rBf", plssvm::kernel_function_type::rbf);
    EXPECT_CONVERSION_FROM_STRING("2", plssvm::kernel_function_type::rbf);
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
}
TEST(KernelType, kernel_to_math_string_unkown) {
    // check conversion from an unknown plssvm::kernel_function_type to the (non-existing) math string
    EXPECT_EQ(plssvm::kernel_function_type_to_math_string(static_cast<plssvm::kernel_function_type>(3)), "unknown");
}

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

// the vector sizes used in the kernel function tests
constexpr std::array kernel_vector_sizes_to_test{ 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
// the kernel type parameter used in the kernel function tests
template <plssvm::kernel_function_type kernel>
constexpr std::array parameter_set{ plssvm::parameter{ kernel, 3, 0.05, 1.0, 1.0 },
                                    plssvm::parameter{ kernel, 1, 0.0, 0.0, 1.0 },
                                    plssvm::parameter{ kernel, 4, -0.05, 1.5, 1.0 },
                                    plssvm::parameter{ kernel, 2, 0.025, -1.0, 1.0 } };

TEST(KernelFunction, linear_kernel_function_variadic) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test the linear kernel function using different parameter sets
        for ([[maybe_unused]] const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::linear>) {
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2), compare::detail::linear_kernel(x1, x2), 256.0);
        }
    }
}

TEST(KernelFunction, linear_kernel_function_parameter) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test the linear kernel function using different parameter sets
        for (const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::linear>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), compare::detail::linear_kernel(x1, x2));
        }
    }
}
TEST(KernelFunction, polynomial_kernel_function_variadic) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test polynomial kernel function
        for (const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::polynomial>) {
            const int degree = params.degree;
            const plssvm::real_type gamma = params.gamma;
            const plssvm::real_type coef0 = params.coef0;
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, degree, gamma, coef0), compare::detail::polynomial_kernel(x1, x2, degree, gamma, coef0), 256.0);
        }
    }
}
TEST(KernelFunction, polynomial_kernel_function_parameter) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test polynomial kernel function
        for (const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::polynomial>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params),
                                       compare::detail::polynomial_kernel(x1, x2, params.degree.value(), params.gamma.value(), params.coef0.value()));
        }
    }
}

TEST(KernelFunction, radial_basis_function_kernel_function_variadic) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test rbf kernel function
        for (const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::rbf>) {
            const plssvm::real_type gamma = params.gamma;
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, gamma), compare::detail::rbf_kernel(x1, x2, gamma), 256.0);
        }
    }
}
TEST(KernelFunction, radial_basis_function_kernel_function_parameter) {
    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<plssvm::real_type> x1 = util::generate_random_vector<plssvm::real_type>(size);
        const std::vector<plssvm::real_type> x2 = util::generate_random_vector<plssvm::real_type>(size);

        // test rbf kernel function
        for (const plssvm::parameter &params : parameter_set<plssvm::kernel_function_type::rbf>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), compare::detail::rbf_kernel(x1, x2, params.gamma.value()));
        }
    }
}

TEST(KernelFunction, unknown_kernel_function_parameter) {
    // create two vectors
    const std::vector<plssvm::real_type> x1 = { plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> x2 = { plssvm::real_type{ 1.0 } };
    // create a parameter object with an unknown kernel type
    plssvm::parameter params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(3);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(x1, x2, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 3)!");
}

TEST(KernelFunctionDeathTest, size_mismatch_kernel_function_variadic) {
    // create random vector with the specified size
    const std::vector<plssvm::real_type> x1{ plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> x2{ plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } };

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, plssvm::real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
}
TEST(KernelFunctionDeathTest, size_mismatch_kernel_function_parameter) {
    // create random vector with the specified size
    const std::vector<plssvm::real_type> x1{ plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> x2{ plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } };

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(x1, x2, plssvm::parameter{}), "Sizes mismatch!: 1 != 2");
}