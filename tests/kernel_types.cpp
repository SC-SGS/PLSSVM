/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different kernel_types.
 */

#include "plssvm/kernel_types.hpp"

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "backends/compare.hpp"  // compare::detail::{linear_kernel, poly_kernel, rbf_kernel}
#include "utility.hpp"           // util::{convert_to_string, convert_from_string, gtest_assert_floating_point_near}, EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH

#include <algorithm>  // std::generate
#include <cstddef>    // std::size_t
#include <random>     // std::random_device, std::mt19937, std::uniform_real_distribution
#include <sstream>    // std::istringstream
#include <vector>     // std::vector

// check whether the plssvm::kernel_type -> std::string conversions are correct
TEST(KernelType, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_type::linear), "linear");
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_type::polynomial), "polynomial");
    EXPECT_EQ(util::convert_to_string(plssvm::kernel_type::rbf), "rbf");
}
TEST(KernelType, to_string_unknown) {
    // check conversions to std::string from unknown kernel_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::kernel_type>(3)), "unknown");
}

// check whether the std::string -> plssvm::kernel_type conversions are correct
TEST(KernelType, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("linear"), plssvm::kernel_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("LINEAR"), plssvm::kernel_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("0"), plssvm::kernel_type::linear);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("polynomial"), plssvm::kernel_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("POLynomIAL"), plssvm::kernel_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("1"), plssvm::kernel_type::polynomial);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("rbf"), plssvm::kernel_type::rbf);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("rBf"), plssvm::kernel_type::rbf);
    EXPECT_EQ(util::convert_from_string<plssvm::kernel_type>("2"), plssvm::kernel_type::rbf);
}
TEST(KernelType, from_string_unknown) {
    // foo isn't a valid kernel_type
    std::istringstream ss{ "foo" };
    plssvm::kernel_type k;
    ss >> k;
    EXPECT_TRUE(ss.fail());
}

TEST(KernelType, kernel_to_math_string) {
    // check conversion from plssvm::kernel_type to the respective math function string
    EXPECT_EQ(plssvm::kernel_type_to_math_string(plssvm::kernel_type::linear), "u'*v");
    EXPECT_EQ(plssvm::kernel_type_to_math_string(plssvm::kernel_type::polynomial), "(gamma*u'*v+coef0)^degree");
    EXPECT_EQ(plssvm::kernel_type_to_math_string(plssvm::kernel_type::rbf), "exp(-gamma*|u-v|^2)");
}

template <typename T>
std::vector<T> generate_random_vector(const std::size_t size) {
    std::vector<T> vec(size);

    // fill vectors with random values
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });

    return vec;
}

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;
const std::vector<std::size_t> kernel_vector_sizes_to_test = { 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };

template <typename T, plssvm::kernel_type kernel>
struct parameter_set {
    std::vector<plssvm::parameter<T>> params = {
        plssvm::parameter<T>{ kernel, 3, 0.05, 1.0, 1.0 },
        plssvm::parameter<T>{ kernel, 1, 0.0, 0.0, 1.0 },
        plssvm::parameter<T>{ kernel, 4, -0.05, 1.5, 1.0 },
        plssvm::parameter<T>{ kernel, 2, 0.025, -1.0, 1.0 }
    };
};

template <typename T>
class KernelTypeBase : public ::testing::Test {};
TYPED_TEST_SUITE(KernelTypeBase, floating_point_types);

TYPED_TEST(KernelTypeBase, linear_kernel_function_variadic) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::linear> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test the linear kernel function using different parameter sets
        for ([[maybe_unused]] const plssvm::parameter<real_type> &p : params_set.params) {
            util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::linear>(x1, x2), compare::detail::linear_kernel(x1, x2));
        }
    }
}

TYPED_TEST(KernelTypeBase, linear_kernel_function_parameter) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::linear> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test the linear kernel function using different parameter sets
        for (const plssvm::parameter<real_type> &p : params_set.params) {
            util::gtest_assert_floating_point_near(plssvm::kernel_function(x1, x2, p), compare::detail::linear_kernel(x1, x2));
        }
    }
}

TYPED_TEST(KernelTypeBase, polynomial_kernel_function_variadic) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::polynomial> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test polynomial kernel function
        for (const plssvm::parameter<real_type> &p : params_set.params) {
            const int degree = p.degree.value();
            const real_type gamma = p.gamma.value();
            const real_type coef0 = p.coef0.value();
            util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::polynomial>(x1, x2, degree, gamma, coef0), compare::detail::poly_kernel(x1, x2, degree, gamma, coef0));
        }
    }
}

TYPED_TEST(KernelTypeBase, polynomial_kernel_function_parameter) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::polynomial> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test polynomial kernel function
        for (const plssvm::parameter<real_type> &p : params_set.params) {
            util::gtest_assert_floating_point_near(plssvm::kernel_function(x1, x2, p), compare::detail::poly_kernel(x1, x2, p.degree.value(), p.gamma.value(), p.coef0.value()));
        }
    }
}

TYPED_TEST(KernelTypeBase, radial_basis_function_kernel_function_variadic) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::rbf> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test rbf kernel function
        for (const plssvm::parameter<real_type> &p : params_set.params) {
            const real_type gamma = p.gamma.value();
            util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::rbf>(x1, x2, gamma), compare::detail::radial_kernel(x1, x2, gamma));
        }
    }
}

TYPED_TEST(KernelTypeBase, radial_basis_function_kernel_function_parameter) {
    using real_type = TypeParam;

    // create parameter set to test
    parameter_set<real_type, plssvm::kernel_type::rbf> params_set;

    for (const std::size_t size : kernel_vector_sizes_to_test) {
        // create random vector with the specified size
        const std::vector<real_type> x1 = generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = generate_random_vector<real_type>(size);

        // test rbf kernel function
        for (const plssvm::parameter<real_type> &p : params_set.params) {
            util::gtest_assert_floating_point_near(plssvm::kernel_function(x1, x2, p), compare::detail::radial_kernel(x1, x2, p.gamma.value()));
        }
    }
}

TYPED_TEST(KernelTypeBase, unknown_kernel_function_parameter) {
    using real_type = TypeParam;

    // create two vectors
    const std::vector<real_type> x1 = { real_type{ 1.0 } };
    const std::vector<real_type> x2 = { real_type{ 1.0 } };
    plssvm::parameter<real_type> p{};
    p.kernel = static_cast<plssvm::kernel_type>(3);

    [[maybe_unused]] real_type res;
    EXPECT_THROW_WHAT(res = plssvm::kernel_function(x1, x2, p), plssvm::unsupported_kernel_type_exception, "Unknown kernel type (value: 3)!");
}

template <typename T>
class KernelTypeBaseDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(KernelTypeBaseDeathTest, floating_point_types);

TYPED_TEST(KernelTypeBaseDeathTest, size_mismatch_kernel_function_variadic) {
    using real_type = TypeParam;

    // create random vector with the specified size
    const std::vector<real_type> x1{ real_type{ 1.0 } };
    const std::vector<real_type> x2{ real_type{ 1.0 }, real_type{ 2.0 } };

    // test mismatched vector sizes
    [[maybe_unused]] real_type res;
    EXPECT_DEATH(res = plssvm::kernel_function<plssvm::kernel_type::linear>(x1, x2), "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(res = plssvm::kernel_function<plssvm::kernel_type::polynomial>(x1, x2, 0, real_type{ 0.0 }, real_type{ 0.0 }), "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(res = plssvm::kernel_function<plssvm::kernel_type::rbf>(x1, x2, real_type{ 0.0 }), "Sizes mismatch!: 1 != 2");
}

TYPED_TEST(KernelTypeBaseDeathTest, size_mismatch_kernel_function_parameter) {
    using real_type = TypeParam;

    // create random vector with the specified size
    const std::vector<real_type> x1{ real_type{ 1.0 } };
    const std::vector<real_type> x2{ real_type{ 1.0 }, real_type{ 2.0 } };

    // test mismatched vector sizes
    [[maybe_unused]] real_type res;
    EXPECT_DEATH(res = plssvm::kernel_function(x1, x2, plssvm::parameter<real_type>{}), "Sizes mismatch!: 1 != 2");
}