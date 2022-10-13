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

#include "backends/compare.hpp"    // compare::detail::{linear_kernel, poly_kernel, rbf_kernel}
#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_NEAR
#include "naming.hpp"              // naming::real_type_to_name
#include "utility.hpp"             // util::{convert_to_string, convert_from_string, util::generate_random_vector, gtest_assert_floating_point_near}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH

#include <algorithm>  // std::generate
#include <array>      // std::array
#include <cstddef>    // std::size_t
#include <sstream>    // std::istringstream
#include <tuple>      // std::ignore
#include <vector>     // std::vector

// check whether the plssvm::kernel_function_type -> std::string conversions are correct
TEST(KernelType, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_function_type::linear), "linear");
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_function_type::polynomial), "polynomial");
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_function_type::rbf), "rbf");
}
TEST(KernelType, to_string_unknown) {
    // check conversions to std::string from unknown kernel_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::kernel_function_type>(3)), "unknown");
}

// check whether the std::string -> plssvm::kernel_function_type conversions are correct
TEST(KernelType, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("linear"), plssvm::kernel_function_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("LINEAR"), plssvm::kernel_function_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("0"), plssvm::kernel_function_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("polynomial"), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("POLynomIAL"), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("1"), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("rbf"), plssvm::kernel_function_type::rbf);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("rBf"), plssvm::kernel_function_type::rbf);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_function_type>("2"), plssvm::kernel_function_type::rbf);
}
TEST(KernelType, from_string_unknown) {
    // foo isn't a valid kernel_type
    std::istringstream input{ "foo" };
    plssvm::kernel_function_type kernel;
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
template <typename T, plssvm::kernel_function_type kernel>
constexpr std::array parameter_set{ plssvm::detail::parameter<T>{ kernel, 3, 0.05, 1.0, 1.0 },
                                    plssvm::detail::parameter<T>{ kernel, 1, 0.0, 0.0, 1.0 },
                                    plssvm::detail::parameter<T>{ kernel, 4, -0.05, 1.5, 1.0 },
                                    plssvm::detail::parameter<T>{ kernel, 2, 0.025, -1.0, 1.0 } };

template <typename T>
class KernelFunction : public ::testing::Test {};
TYPED_TEST_SUITE(KernelFunction, floating_point_types, naming::real_type_to_name);

TYPED_TEST(KernelFunction, linear_kernel_function_variadic) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test the linear kernel function using different parameter sets
        for ([[maybe_unused]] const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::linear>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2), compare::detail::linear_kernel(x1, x2));
        }
    }
}

TYPED_TEST(KernelFunction, linear_kernel_function_parameter) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test the linear kernel function using different parameter sets
        for (const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::linear>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), compare::detail::linear_kernel(x1, x2));
        }
    }
}
TYPED_TEST(KernelFunction, polynomial_kernel_function_variadic) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test polynomial kernel function
        for (const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::polynomial>) {
            const int degree = params.degree;
            const real_type gamma = params.gamma;
            const real_type coef0 = params.coef0;
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, degree, gamma, coef0), compare::detail::poly_kernel(x1, x2, degree, gamma, coef0));
        }
    }
}
TYPED_TEST(KernelFunction, polynomial_kernel_function_parameter) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test polynomial kernel function
        for (const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::polynomial>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params),
                                       compare::detail::poly_kernel(x1, x2, params.degree.value(), params.gamma.value(), params.coef0.value()));
        }
    }
}

TYPED_TEST(KernelFunction, radial_basis_function_kernel_function_variadic) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test rbf kernel function
        for (const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::rbf>) {
            const real_type gamma = params.gamma;
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, gamma), compare::detail::radial_kernel(x1, x2, gamma));
        }
    }
}
TYPED_TEST(KernelFunction, radial_basis_function_kernel_function_parameter) {
    using real_type = TypeParam;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        // test rbf kernel function
        for (const plssvm::detail::parameter<real_type> &params : parameter_set<real_type, plssvm::kernel_function_type::rbf>) {
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), compare::detail::radial_kernel(x1, x2, params.gamma.value()));
        }
    }
}

TYPED_TEST(KernelFunction, unknown_kernel_function_parameter) {
    using real_type = TypeParam;

    // create two vectors
    const std::vector<real_type> x1 = { real_type{ 1.0 } };
    const std::vector<real_type> x2 = { real_type{ 1.0 } };
    // create a parameter object with an unknown kernel type
    plssvm::detail::parameter<real_type> params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(3);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(x1, x2, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 3)!");
}

template <typename T>
using KernelFunctionDeathTest = KernelFunction<T>;
TYPED_TEST_SUITE(KernelFunctionDeathTest, floating_point_types, naming::real_type_to_name);

TYPED_TEST(KernelFunctionDeathTest, size_mismatch_kernel_function_variadic) {
    using real_type = TypeParam;

    // create random vector with the specified size
    const std::vector<real_type> x1{ real_type{ 1.0 } };
    const std::vector<real_type> x2{ real_type{ 1.0 }, real_type{ 2.0 } };

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
}
TYPED_TEST(KernelFunctionDeathTest, size_mismatch_kernel_function_parameter) {
    using real_type = TypeParam;

    // create random vector with the specified size
    const std::vector<real_type> x1{ real_type{ 1.0 } };
    const std::vector<real_type> x2{ real_type{ 1.0 }, real_type{ 2.0 } };

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(x1, x2, plssvm::detail::parameter<real_type>{}), "Sizes mismatch!: 1 != 2");
}