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

#include "plssvm/constants.hpp"       // plssvm::PADDING_SIZE
#include "plssvm/detail/utility.hpp"  // plssvm::detail::{contains, erase_if}
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "backends/compare.hpp"    // compare::detail::{linear_kernel, poly_kernel, rbf_kernel}
#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_NEAR, EXPECT_FLOATING_POINT_NEAR_EPS
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{real_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::{generate_random_vector, generate_random_matrix}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, SCOPED_TRACE, ::testing::Test

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <sstream>  // std::istringstream
#include <tuple>    // std::tuple
#include <vector>   // std::vector

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
    EXPECT_CONVERSION_FROM_STRING("poly", plssvm::kernel_function_type::polynomial);
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

//*************************************************************************************************************************************//
//                                                  kernel functions using std::vector                                                 //
//*************************************************************************************************************************************//

template <typename T>
class KernelFunctionVector : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;

    /**
     * @brief Return the different vector sizes to test.
     * @return the vector sizes (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<std::size_t> get_sizes() const noexcept { return sizes_; }
    /**
     * @brief Return the different parameter values to test.
     * @return the parameter values (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::array<plssvm::real_type, 4>> &get_param_values() const noexcept { return param_values_; }

  private:
    /// The different vector sizes to test.
    std::vector<std::size_t> sizes_{ 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 };
    /// The different parameter values to test.
    std::vector<std::array<plssvm::real_type, 4>> param_values_{
        std::array{ plssvm::real_type{ 3.0 }, plssvm::real_type{ 0.05 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
        std::array{ plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } },
        std::array{ plssvm::real_type{ 4.0 }, plssvm::real_type{ -0.05 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } },
        std::array{ plssvm::real_type{ 2.0 }, plssvm::real_type{ 0.025 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 0.5 } },
    };
};
TYPED_TEST_SUITE(KernelFunctionVector, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(KernelFunctionVector, linear_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2), compare::detail::linear_kernel(x1, x2), real_type{ 512.0 });
        }
    }
}
TYPED_TEST(KernelFunctionVector, linear_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::linear, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(x1, x2, params), compare::detail::linear_kernel(x1, x2), real_type{ 512.0 });
        }
    }
}

TYPED_TEST(KernelFunctionVector, polynomial_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const auto degree_p = static_cast<int>(degree);
            const auto gamma_p = static_cast<real_type>(gamma);
            const auto coef0_p = static_cast<real_type>(coef0);
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, degree_p, gamma_p, coef0_p), compare::detail::polynomial_kernel(x1, x2, degree_p, gamma_p, coef0_p), real_type{ 512.0 });
        }
    }
}
TYPED_TEST(KernelFunctionVector, polynomial_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(x1, x2, params),
                                           compare::detail::polynomial_kernel(x1, x2, params.degree.value(), static_cast<real_type>(params.gamma.value()), static_cast<real_type>(params.coef0.value())),
                                           real_type{ 512.0 });
        }
    }
}

TYPED_TEST(KernelFunctionVector, radial_basis_function_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const auto gamma_p = static_cast<real_type>(gamma);
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, gamma_p), compare::detail::rbf_kernel(x1, x2, gamma_p), real_type{ 512.0 });
        }
    }
}
TYPED_TEST(KernelFunctionVector, radial_basis_function_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::rbf, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(x1, x2, params), compare::detail::rbf_kernel(x1, x2, static_cast<real_type>(params.gamma.value())), real_type{ 512.0 });
        }
    }
}

TYPED_TEST(KernelFunctionVector, unknown_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    // create two vectors
    const std::vector<real_type> x1 = { real_type{ 1.0 } };
    const std::vector<real_type> x2 = { real_type{ 1.0 } };
    // create a parameter object with an unknown kernel type
    plssvm::parameter params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(3);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(x1, x2, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 3)!");
}

template <typename T>
class KernelFunctionVectorDeathTest : public KernelFunctionVector<T> {};
TYPED_TEST_SUITE(KernelFunctionVectorDeathTest, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(KernelFunctionVectorDeathTest, size_mismatch_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

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
TYPED_TEST(KernelFunctionVectorDeathTest, size_mismatch_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    // create random vector with the specified size
    const std::vector<real_type> x1{ real_type{ 1.0 } };
    const std::vector<real_type> x2{ real_type{ 1.0 }, real_type{ 2.0 } };

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(x1, x2, plssvm::parameter{}), "Sizes mismatch!: 1 != 2");
}

//*************************************************************************************************************************************//
//                                                kernel functions using plssvm::matrix                                                //
//*************************************************************************************************************************************//

template <typename T>
class KernelFunctionMatrix : public KernelFunctionVector<T> {
  protected:
    using typename KernelFunctionVector<T>::fixture_real_type;
    static constexpr plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;

    /**
     * @brief Return the different sizes to test. Removes the zero entry since it is not applicable to the matrix tests.
     * @return the sizes (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::size_t> get_sizes() const noexcept override {
        // get the sizes defined by the base class
        std::vector<std::size_t> sizes = KernelFunctionVector<T>::get_sizes();
        // erase the size 0 entry
        plssvm::detail::erase_if(sizes, [](const std::size_t size) { return size == 0; });
        return sizes;
    }
};
TYPED_TEST_SUITE(KernelFunctionMatrix, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(KernelFunctionMatrix, linear_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, i, matr2, j), compare::detail::linear_kernel(x1, x2), real_type{ 512.0 });
                }
            }
        }
    }
}
TYPED_TEST(KernelFunctionMatrix, linear_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    const plssvm::parameter params{ plssvm::kernel_function_type::linear, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(matr1, i, matr2, j, params), compare::detail::linear_kernel(x1, x2), real_type{ 512.0 });
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, polynomial_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    const auto degree_p = static_cast<int>(degree);
                    const auto gamma_p = static_cast<real_type>(gamma);
                    const auto coef0_p = static_cast<real_type>(coef0);
                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, i, matr2, j, degree_p, gamma_p, coef0_p), compare::detail::polynomial_kernel(x1, x2, degree_p, gamma_p, coef0_p), real_type{ 512.0 });
                }
            }
        }
    }
}
TYPED_TEST(KernelFunctionMatrix, polynomial_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(matr1, i, matr2, j, params),
                                                   compare::detail::polynomial_kernel(x1, x2, params.degree.value(), static_cast<real_type>(params.gamma.value()), static_cast<real_type>(params.coef0.value())),
                                                   real_type{ 512.0 });
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, rbf_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    const auto gamma_p = static_cast<real_type>(gamma);
                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, i, matr2, j, gamma_p), compare::detail::rbf_kernel(x1, x2, gamma_p), real_type{ 512.0 });
                }
            }
        }
    }
}
TYPED_TEST(KernelFunctionMatrix, rbf_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(4, size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            ASSERT_EQ(matr1.shape(), matr2.shape());

            for (std::size_t i = 0; i < matr1.num_rows(); ++i) {
                for (std::size_t j = 0; j < matr1.num_rows(); ++j) {
                    SCOPED_TRACE(fmt::format("i: {}; j: {}", i, j));

                    // create vectors for ground truth calculation
                    std::vector<real_type> x1(matr1.num_cols());
                    std::vector<real_type> x2(matr2.num_cols());
                    for (std::size_t dim = 0; dim < matr1.num_cols(); ++dim) {
                        x1[dim] = matr1(i, dim);
                        x2[dim] = matr2(j, dim);
                    }

                    const plssvm::parameter params{ plssvm::kernel_function_type::rbf, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR_EPS(plssvm::kernel_function(matr1, i, matr2, j, params),
                                                   compare::detail::rbf_kernel(x1, x2, static_cast<real_type>(params.gamma.value())),
                                                   real_type{ 512.0 });
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, unknown_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2);
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create a parameter object with an unknown kernel type
    plssvm::parameter params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(3);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 0, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 3)!");
}

template <typename T>
class KernelFunctionMatrixDeathTest : public KernelFunctionMatrix<T> {};
TYPED_TEST_SUITE(KernelFunctionMatrixDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(KernelFunctionMatrixDeathTest, size_mismatch_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 1);
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, 0, matr2, 0),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, 0, matr2, 0, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, 0, matr2, 0, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
}
TYPED_TEST(KernelFunctionMatrixDeathTest, invalid_indices_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2);
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // test mismatched vector indices
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, 0, matr2, 2),
                 "Out-of-bounce access for j and y!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, 0, matr2, 3, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Out-of-bounce access for j and y!: 3 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, 2, matr2, 0, real_type{ 0.0 }),
                 "Out-of-bounce access for i and x!: 2 < 2");
}

TYPED_TEST(KernelFunctionMatrixDeathTest, size_mismatch_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 1);
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 0, plssvm::parameter{}), "Sizes mismatch!: 1 != 2");
}
TYPED_TEST(KernelFunctionMatrixDeathTest, invalid_indices_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2);
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(2, 2, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // test mismatched vector indices
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 2, plssvm::parameter{}),
                 "Out-of-bounce access for j and y!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 3, plssvm::parameter{}),
                 "Out-of-bounce access for j and y!: 3 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 2, matr2, 0, plssvm::parameter{}),
                 "Out-of-bounce access for i and x!: 2 < 2");
}