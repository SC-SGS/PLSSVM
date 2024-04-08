/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the std::vector arithmetic operator overloads.
 */

#include "plssvm/detail/operators.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ, EXPECT_FLOATING_POINT_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{real_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}

#include "gtest/gtest.h"  // TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_DEATH, ::testing::{Types, Test}

#include <tuple>   // std::ignore
#include <vector>  // std::vector

// make all operator overloads available in all tests
using namespace plssvm::operators;

//*************************************************************************************************************************************//
//                                                          scalar operations                                                          //
//*************************************************************************************************************************************//
template <typename T>
class ScalarOperations : public ::testing::Test { };

TYPED_TEST_SUITE(ScalarOperations, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ScalarOperations, operator_sign_positive) {
    using real_type = util::test_parameter_type_at_t<0, TypeParam>;

    EXPECT_FLOATING_POINT_EQ(sign(real_type{ 1.6 }), real_type{ 1 });
    EXPECT_FLOATING_POINT_EQ(sign(real_type{ 3 }), real_type{ 1 });
}

TYPED_TEST(ScalarOperations, operator_sign_negative) {
    using real_type = util::test_parameter_type_at_t<0, TypeParam>;

    EXPECT_FLOATING_POINT_EQ(sign(real_type{ -2.4 }), real_type{ -1 });
    EXPECT_FLOATING_POINT_EQ(sign(real_type{ -4 }), real_type{ -1 });
    EXPECT_FLOATING_POINT_EQ(sign(real_type{ 0 }), real_type{ -1 });
}

//*************************************************************************************************************************************//
//                                                        std::vector operations                                                       //
//*************************************************************************************************************************************//
template <typename T>
class VectorOperations : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        a_ = { 1, 2, 3, 4, 5 };
        b_ = { 1.5, 2.5, 3.5, 4.5, 5.5 };
        scalar_ = 1.5;
    }

    /**
     * @brief Return the sample vector @p a.
     * @return the sample vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_a() noexcept { return a_; }

    /**
     * @brief Return the sample vector @p b.
     * @return the sample vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_b() noexcept { return b_; }

    /**
     * @brief Return the empty vector.
     * @return the empty vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_empty() noexcept { return empty_; }

    /**
     * @brief Return the sample scalar.
     * @return the scalar (`[[nodiscard]]`)
     */
    [[nodiscard]] fixture_real_type &get_scalar() noexcept { return scalar_; }

  private:
    /// Sample vector to test the different operations.
    std::vector<fixture_real_type> a_{};
    /// Sample vector to test the different operations.
    std::vector<fixture_real_type> b_{};
    /// Empty vector to test the different operations.
    std::vector<fixture_real_type> empty_{};
    /// Sample scalar to test the different operations.
    fixture_real_type scalar_{};
};

TYPED_TEST_SUITE(VectorOperations, util::real_type_gtest, naming::test_parameter_to_name);

template <typename T>
class VectorOperationsDeathTest : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        a_ = { 1, 2, 3, 4 };
        b_ = { 5, 6 };
    }

    /**
     * @brief Return the sample vector @p a.
     * @return the sample vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_a() noexcept { return a_; }

    /**
     * @brief Return the sample vector @p b.
     * @return the sample vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_b() noexcept { return b_; }

  private:
    /// Sample vector to test the different operations.
    std::vector<fixture_real_type> a_{};
    /// Sample vector to test the different operations.
    std::vector<fixture_real_type> b_{};
};

TYPED_TEST_SUITE(VectorOperationsDeathTest, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(VectorOperations, operator_add_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary addition using two vectors
    const std::vector<real_type> c = { 2.5, 4.5, 6.5, 8.5, 10.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() + this->get_b(), c);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_b() + this->get_a(), c);
}

TYPED_TEST(VectorOperations, operator_add_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound addition using two vectors
    const std::vector<real_type> c = { 2.5, 4.5, 6.5, 8.5, 10.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() += this->get_b(), c);
}

TYPED_TEST(VectorOperations, operator_add_scalar_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary addition using a vector and a scalar
    const std::vector<real_type> c = { 2.5, 3.5, 4.5, 5.5, 6.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() + this->get_scalar(), c);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_scalar() + this->get_a(), c);
}

TYPED_TEST(VectorOperations, operator_add_scalar_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound addition using a vector and a scalar
    const std::vector<real_type> c = { 2.5, 3.5, 4.5, 5.5, 6.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() += this->get_scalar(), c);
}

TYPED_TEST(VectorOperations, operator_add_binary_empty) {
    // binary addition using two empty vectors
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_empty() + this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_add_compound_empty) {
    // compound addition using two empty vectors
    EXPECT_EQ(this->get_empty() += this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_add_scalar_binary_empty) {
    // binary addition using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() + this->get_scalar(), this->get_empty());
    EXPECT_EQ(this->get_scalar() + this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_add_scalar_compound_empty) {
    // compound addition using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() += this->get_scalar(), this->get_empty());
}

TYPED_TEST(VectorOperationsDeathTest, operator_add_binary) {
    // try to binary add vectors with different sizes
    EXPECT_DEATH(std::ignore = this->get_a() + this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = this->get_b() + this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperationsDeathTest, operator_add_compound) {
    // try to compound add vectors with different sizes
    EXPECT_DEATH(this->get_a() += this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->get_b() += this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_subtract_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary subtraction using two vectors
    {
        const std::vector<real_type> c = { -0.5, -0.5, -0.5, -0.5, -0.5 };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() - this->get_b(), c);
    }
    {
        const std::vector<real_type> c = { 0.5, 0.5, 0.5, 0.5, 0.5 };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_b() - this->get_a(), c);
    }
}

TYPED_TEST(VectorOperations, operator_subtract_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound subtraction using two vectors
    const std::vector<real_type> c = { -0.5, -0.5, -0.5, -0.5, -0.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() -= this->get_b(), c);
}

TYPED_TEST(VectorOperations, operator_subtract_scalar_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary subtraction using a vector and a scalar
    {
        const std::vector<real_type> c = { -0.5, 0.5, 1.5, 2.5, 3.5 };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() - this->get_scalar(), c);
    }
    {
        const std::vector<real_type> c = { 0.5, -0.5, -1.5, -2.5, -3.5 };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_scalar() - this->get_a(), c);
    }
}

TYPED_TEST(VectorOperations, operator_subtract_scalar_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound subtraction using a vector and a scalar
    const std::vector<real_type> c = { -0.5, 0.5, 1.5, 2.5, 3.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() -= this->get_scalar(), c);
}

TYPED_TEST(VectorOperations, operator_subtract_binary_empty) {
    // binary subtraction using two empty vectors
    EXPECT_EQ(this->get_empty() - this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_subtract_compound_empty) {
    // compound subtraction using two empty vectors
    EXPECT_EQ(this->get_empty() -= this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_subtract_scalar_binary_empty) {
    // binary subtraction using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() - this->get_scalar(), this->get_empty());
    EXPECT_EQ(this->get_scalar() - this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_subtract_scalar_compound_empty) {
    // compound subtraction using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() -= this->get_scalar(), this->get_empty());
}

TYPED_TEST(VectorOperationsDeathTest, operator_subtract_binary) {
    // try to binary subtract vectors with different sizes
    EXPECT_DEATH(std::ignore = this->get_a() - this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = this->get_b() - this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperationsDeathTest, operator_subtract_compound) {
    // try to compound subtract vectors with different sizes
    EXPECT_DEATH(this->get_a() -= this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->get_b() -= this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_multiply_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary multiplication using two vectors
    const std::vector<real_type> c = { 1.5, 5, 10.5, 18, 27.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() * this->get_b(), c);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_b() * this->get_a(), c);
}

TYPED_TEST(VectorOperations, operator_multiply_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound multiplication using two vectors
    const std::vector<real_type> c = { 1.5, 5, 10.5, 18, 27.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() *= this->get_b(), c);
}

TYPED_TEST(VectorOperations, operator_multiply_scalar_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary multiplication using a vector and a scalar
    const std::vector<real_type> c = { 1.5, 3, 4.5, 6, 7.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() * this->get_scalar(), c);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_scalar() * this->get_a(), c);
}

TYPED_TEST(VectorOperations, operator_multiply_scalar_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound multiplication using a vector and a scalar
    const std::vector<real_type> c = { 1.5, 3, 4.5, 6, 7.5 };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() *= this->get_scalar(), c);
}

TYPED_TEST(VectorOperations, operator_multiply_binary_empty) {
    // binary multiplication using two empty vectors
    EXPECT_EQ(this->get_empty() * this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_multiply_compound_empty) {
    // compound multiplication using two empty vectors
    EXPECT_EQ(this->get_empty() *= this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_multiply_scalar_binary_empty) {
    // binary multiplication using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() * this->get_scalar(), this->get_empty());
    EXPECT_EQ(this->get_scalar() * this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_multiply_scalar_compound_empty) {
    // compound multiplication using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() *= this->get_scalar(), this->get_empty());
}

TYPED_TEST(VectorOperationsDeathTest, operator_multiply_binary) {
    // try to binary multiply vectors with different sizes
    EXPECT_DEATH(std::ignore = this->get_a() * this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = this->get_b() * this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperationsDeathTest, operator_multiply_compound) {
    // try to compound multiply vectors with different sizes
    EXPECT_DEATH(this->get_a() *= this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->get_b() *= this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_divide_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary division using two vectors
    {
        const std::vector<real_type> c = { real_type{ 1.0 / 1.5 }, real_type{ 2.0 / 2.5 }, real_type{ 3.0 / 3.5 }, real_type{ 4.0 / 4.5 }, real_type{ 5.0 / 5.5 } };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() / this->get_b(), c);
    }
    {
        const std::vector<real_type> c = { 1.5, real_type{ 2.5 / 2.0 }, real_type{ 3.5 / 3.0 }, real_type{ 4.5 / 4.0 }, real_type{ 5.5 / 5.0 } };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_b() / this->get_a(), c);
    }
}

TYPED_TEST(VectorOperations, operator_divide_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound division using two vectors
    const std::vector<real_type> c = { real_type{ 1.0 / 1.5 }, real_type{ 2.0 / 2.5 }, real_type{ 3.0 / 3.5 }, real_type{ 4.0 / 4.5 }, real_type{ 5.0 / 5.5 } };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() /= this->get_b(), c);
}

TYPED_TEST(VectorOperations, operator_divide_scalar_binary) {
    using real_type = typename TestFixture::fixture_real_type;

    // binary division using a vector and a scalar
    {
        const std::vector<real_type> c = { real_type{ 1.0 / 1.5 }, real_type{ 2.0 / 1.5 }, 2.0, real_type{ 4.0 / 1.5 }, real_type{ 5.0 / 1.5 } };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() / this->get_scalar(), c);
    }
    {
        const std::vector<real_type> c = { 1.5, real_type{ 1.5 / 2.0 }, 0.5, real_type{ 1.5 / 4.0 }, real_type{ 1.5 / 5.0 } };
        EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_scalar() / this->get_a(), c);
    }
}

TYPED_TEST(VectorOperations, operator_divide_scalar_compound) {
    using real_type = typename TestFixture::fixture_real_type;

    // compound division using a vector and a scalar
    const std::vector<real_type> c = { real_type{ 1.0 / 1.5 }, real_type{ 2.0 / 1.5 }, 2.0, real_type{ 4.0 / 1.5 }, real_type{ 5.0 / 1.5 } };
    EXPECT_FLOATING_POINT_VECTOR_NEAR(this->get_a() /= this->get_scalar(), c);
}

TYPED_TEST(VectorOperations, operator_divide_binary_empty) {
    // binary division using two empty vectors
    EXPECT_EQ(this->get_empty() / this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_divide_compound_empty) {
    // compound division using two empty vectors
    EXPECT_EQ(this->get_empty() /= this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_divide_scalar_binary_empty) {
    // binary division using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() / this->get_scalar(), this->get_empty());
    EXPECT_EQ(this->get_scalar() / this->get_empty(), this->get_empty());
}

TYPED_TEST(VectorOperations, operator_divide_scalar_compound_empty) {
    // compound division using an empty vector and a scalar
    EXPECT_EQ(this->get_empty() /= this->get_scalar(), this->get_empty());
}

TYPED_TEST(VectorOperationsDeathTest, operator_divide_binary) {
    // try to binary division vectors with different sizes
    EXPECT_DEATH(std::ignore = this->get_a() / this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = this->get_b() / this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperationsDeathTest, operator_divide_compound) {
    // try to compound division vectors with different sizes
    EXPECT_DEATH(this->get_a() /= this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->get_b() /= this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_dot_function) {
    // calculate dot product using the dot function
    EXPECT_FLOATING_POINT_NEAR(dot(this->get_a(), this->get_b()), 62.5);
    EXPECT_FLOATING_POINT_NEAR(dot(this->get_b(), this->get_a()), 62.5);
}

TYPED_TEST(VectorOperations, operator_dot_transposed) {
    // calculate dot product using the transposed overload function
    EXPECT_FLOATING_POINT_NEAR(transposed{ this->get_a() } * this->get_b(), 62.5);
    EXPECT_FLOATING_POINT_NEAR(transposed{ this->get_b() } * this->get_a(), 62.5);
}

TYPED_TEST(VectorOperationsDeathTest, operator_dot_function) {
    // try to calculate the dot product with vectors of different sizes
    EXPECT_DEATH(std::ignore = dot(this->get_a(), this->get_b()), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = dot(this->get_b(), this->get_a()), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperationsDeathTest, operator_dot_transposed) {
    // try to calculate the dot product with vectors of different sizes
    EXPECT_DEATH(std::ignore = transposed{ this->get_a() } * this->get_b(), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = transposed{ this->get_b() } * this->get_a(), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_sum) {
    // sum vector elements
    EXPECT_FLOATING_POINT_NEAR(sum(this->get_a()), 15);
    EXPECT_FLOATING_POINT_NEAR(sum(this->get_b()), 17.5);
}

TYPED_TEST(VectorOperations, operator_squared_euclidean_dist) {
    // calculate the squared Euclidean distance between two vectors
    EXPECT_FLOATING_POINT_NEAR(squared_euclidean_dist(this->get_a(), this->get_b()), 1.25);
}

TYPED_TEST(VectorOperationsDeathTest, operator_squared_euclidean_dist) {
    // try to calculate the squared Euclidean distance between two vectors with different distance
    EXPECT_DEATH(std::ignore = squared_euclidean_dist(this->get_a(), this->get_b()), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = squared_euclidean_dist(this->get_b(), this->get_a()), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(VectorOperations, operator_manhattan_dist) {
    // calculate the Manhattan distance between two vectors
    EXPECT_FLOATING_POINT_NEAR(manhattan_dist(this->get_a(), this->get_b()), 2.5);
}

TYPED_TEST(VectorOperationsDeathTest, operator_manhattan_dist) {
    // try to calculate the Manhattan distance between two vectors with different distance
    EXPECT_DEATH(std::ignore = manhattan_dist(this->get_a(), this->get_b()), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = manhattan_dist(this->get_b(), this->get_a()), "Sizes mismatch!: 2 != 4");
}