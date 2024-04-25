/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different kernel_types.
 */

#include "plssvm/kernel_functions.hpp"

#include "plssvm/constants.hpp"              // plssvm::PADDING_SIZE
#include "plssvm/detail/utility.hpp"         // plssvm::detail::{contains, erase_if}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_function
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "tests/backends/ground_truth.hpp"  // ground_truth::detail::{linear_kernel, poly_kernel, rbf_kernel}
#include "tests/custom_test_macros.hpp"     // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_NEAR
#include "tests/naming.hpp"                 // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"          // util::{real_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"                // util::{generate_random_vector, generate_random_matrix}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, SCOPED_TRACE, ::testing::Test

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <tuple>    // std::tuple, std::ignore
#include <utility>  // std::pair
#include <variant>  // std::get
#include <vector>   // std::vector

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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::linear>(x1, x2), ground_truth::detail::linear_kernel(x1, x2));
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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), ground_truth::detail::linear_kernel(x1, x2));
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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x1, x2, degree_p, gamma_p, coef0_p), ground_truth::detail::polynomial_kernel(x1, x2, degree_p, gamma_p, coef0_p));
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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params),
                                       ground_truth::detail::polynomial_kernel(x1, x2, params.degree, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma)), static_cast<real_type>(params.coef0)));
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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x1, x2, gamma_p), ground_truth::detail::rbf_kernel(x1, x2, gamma_p));
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
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), ground_truth::detail::rbf_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
        }
    }
}

TYPED_TEST(KernelFunctionVector, sigmoid_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const auto gamma_p = static_cast<real_type>(gamma);
            const auto coef0_p = static_cast<real_type>(coef0);
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(x1, x2, gamma_p, coef0_p), ground_truth::detail::sigmoid_kernel(x1, x2, gamma_p, coef0_p));
        }
    }
}

TYPED_TEST(KernelFunctionVector, sigmoid_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::sigmoid, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), ground_truth::detail::sigmoid_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma)), static_cast<real_type>(params.coef0)));
        }
    }
}

TYPED_TEST(KernelFunctionVector, laplacian_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const auto gamma_p = static_cast<real_type>(gamma);
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(x1, x2, gamma_p), ground_truth::detail::laplacian_kernel(x1, x2, gamma_p));
        }
    }
}

TYPED_TEST(KernelFunctionVector, laplacian_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size);
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size);

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::laplacian, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), ground_truth::detail::laplacian_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
        }
    }
}

TYPED_TEST(KernelFunctionVector, chi_squared_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size, { real_type{ 0.0 }, real_type{ 1.0 } });
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size, { real_type{ 0.0 }, real_type{ 1.0 } });

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const auto gamma_p = static_cast<real_type>(gamma);
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(x1, x2, gamma_p), ground_truth::detail::chi_squared_kernel(x1, x2, gamma_p));
        }
    }
}

TYPED_TEST(KernelFunctionVector, chi_squared_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random vector with the specified size
        const std::vector<real_type> x1 = util::generate_random_vector<real_type>(size, { real_type{ 0.0 }, real_type{ 1.0 } });
        const std::vector<real_type> x2 = util::generate_random_vector<real_type>(size, { real_type{ 0.0 }, real_type{ 1.0 } });

        for (const auto [degree, gamma, coef0, cost] : this->get_param_values()) {
            SCOPED_TRACE(fmt::format("parameter: [{}, {}, {}, {}]", degree, gamma, coef0, cost));
            const plssvm::parameter params{ plssvm::kernel_function_type::chi_squared, static_cast<int>(degree), gamma, coef0, cost };
            EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(x1, x2, params), ground_truth::detail::chi_squared_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
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
    params.kernel_type = static_cast<plssvm::kernel_function_type>(6);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(x1, x2, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 6)!");
}

template <typename T>
class KernelFunctionVectorDeathTest : public KernelFunctionVector<T> { };

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
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(x1, x2, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(x1, x2, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(x1, x2, real_type{ 0.0 }),
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
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;

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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

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

                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, i, matr2, j), ground_truth::detail::linear_kernel(x1, x2));
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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params), ground_truth::detail::linear_kernel(x1, x2));
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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, i, matr2, j, degree_p, gamma_p, coef0_p), ground_truth::detail::polynomial_kernel(x1, x2, degree_p, gamma_p, coef0_p));
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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params),
                                               ground_truth::detail::polynomial_kernel(x1, x2, params.degree, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma)), static_cast<real_type>(params.coef0)));
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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, i, matr2, j, gamma_p), ground_truth::detail::rbf_kernel(x1, x2, gamma_p));
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
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params),
                                               ground_truth::detail::rbf_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, sigmoid_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

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
                    const auto coef0_p = static_cast<real_type>(coef0);
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(matr1, i, matr2, j, gamma_p, coef0_p), ground_truth::detail::sigmoid_kernel(x1, x2, gamma_p, coef0_p));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, sigmoid_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });

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

                    const plssvm::parameter params{ plssvm::kernel_function_type::sigmoid, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params),
                                               ground_truth::detail::sigmoid_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma)), static_cast<real_type>(params.coef0)));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, laplacian_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(matr1, i, matr2, j, gamma_p), ground_truth::detail::laplacian_kernel(x1, x2, gamma_p));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, laplacian_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size });

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

                    const plssvm::parameter params{ plssvm::kernel_function_type::laplacian, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params),
                                               ground_truth::detail::laplacian_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, chi_squared_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, std::pair{ real_type{ 0.0 }, real_type{ 1.0 } });
        auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, std::pair{ real_type{ 0.0 }, real_type{ 1.0 } });

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
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(matr1, i, matr2, j, gamma_p), ground_truth::detail::chi_squared_kernel(x1, x2, gamma_p));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, chi_squared_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    for (const std::size_t size : this->get_sizes()) {
        SCOPED_TRACE(fmt::format("size: {}", size));
        // create random matrices with the specified size
        const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }, std::pair{ real_type{ 0.0 }, real_type{ 1.0 } });
        const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 4, size }, std::pair{ real_type{ 0.0 }, real_type{ 1.0 } });

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

                    const plssvm::parameter params{ plssvm::kernel_function_type::chi_squared, static_cast<int>(degree), gamma, coef0, cost };
                    EXPECT_FLOATING_POINT_NEAR(plssvm::kernel_function(matr1, i, matr2, j, params),
                                               ground_truth::detail::chi_squared_kernel(x1, x2, static_cast<real_type>(std::get<plssvm::real_type>(params.gamma))));
                }
            }
        }
    }
}

TYPED_TEST(KernelFunctionMatrix, unknown_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 });
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // create a parameter object with an unknown kernel type
    plssvm::parameter params{};
    params.kernel_type = static_cast<plssvm::kernel_function_type>(6);

    // using an unknown kernel type must throw
    EXPECT_THROW_WHAT(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 0, params),
                      plssvm::unsupported_kernel_type_exception,
                      "Unknown kernel type (value: 6)!");
}

template <typename T>
class KernelFunctionMatrixDeathTest : public KernelFunctionMatrix<T> { };

TYPED_TEST_SUITE(KernelFunctionMatrixDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(KernelFunctionMatrixDeathTest, size_mismatch_kernel_function_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 1 });
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, 0, matr2, 0),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, 0, matr2, 0, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, 0, matr2, 0, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(matr1, 0, matr2, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(matr1, 0, matr2, 0, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(matr1, 0, matr2, 0, real_type{ 0.0 }),
                 "Sizes mismatch!: 1 != 2");
}

TYPED_TEST(KernelFunctionMatrixDeathTest, invalid_indices_variadic) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 });
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // test mismatched vector indices
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::linear>(matr1, 0, matr2, 2),
                 "Out-of-bounce access for j and y!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(matr1, 0, matr2, 3, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Out-of-bounce access for j and y!: 3 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::rbf>(matr1, 2, matr2, 0, real_type{ 0.0 }),
                 "Out-of-bounce access for i and x!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(matr1, 2, matr2, 0, real_type{ 0.0 }, real_type{ 0.0 }),
                 "Out-of-bounce access for i and x!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(matr1, 2, matr2, 0, real_type{ 0.0 }),
                 "Out-of-bounce access for i and x!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(matr1, 2, matr2, 0, real_type{ 0.0 }),
                 "Out-of-bounce access for i and x!: 2 < 2");
}

TYPED_TEST(KernelFunctionMatrixDeathTest, size_mismatch_kernel_function_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 1 });
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // test mismatched vector sizes
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 0, plssvm::parameter{}), "Sizes mismatch!: 1 != 2");
}

TYPED_TEST(KernelFunctionMatrixDeathTest, invalid_indices_parameter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create two matrices
    const auto matr1 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 });
    const auto matr2 = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 2, 2 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // test mismatched vector indices
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 2, plssvm::parameter{}),
                 "Out-of-bounce access for j and y!: 2 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 0, matr2, 3, plssvm::parameter{}),
                 "Out-of-bounce access for j and y!: 3 < 2");
    EXPECT_DEATH(std::ignore = plssvm::kernel_function(matr1, 2, matr2, 0, plssvm::parameter{}),
                 "Out-of-bounce access for i and x!: 2 < 2");
}
