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

#include "../naming.hpp"         // naming::arithmetic_types_to_name
#include "../types_to_test.hpp"  // util::real_type_gtest

#include "gtest/gtest.h"  // TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_DEATH, ::testing::{Types, Test}

#include <tuple>   // std::ignore
#include <vector>  // std::vector

// make all operator overloads available in all tests
using namespace plssvm::operators;

// testsuite for "normal" tests
template <typename T>
class Operators : public ::testing::Test {
  protected:
    void SetUp() override {
        a = { 1, 2, 3, 4, 5 };
        b = { 1.5, 2.5, 3.5, 4.5, 5.5 };
        scalar = 1.5;
    }

    using real_type = T;

    std::vector<real_type> a{};
    std::vector<real_type> b{};
    std::vector<real_type> empty{};
    real_type scalar{};
};
TYPED_TEST_SUITE(Operators, util::real_type_gtest, naming::real_type_to_name);

// testsuite for death tests
template <typename T>
class OperatorsDeathTest : public ::testing::Test {
  protected:
    void SetUp() override {
        a = { 1, 2, 3, 4 };
        b = { 5, 6 };
    }

    using real_type = T;

    std::vector<real_type> a{};
    std::vector<real_type> b{};
};
TYPED_TEST_SUITE(OperatorsDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(Operators, operator_add_binary) {
    // binary addition using two vectors
    const std::vector<typename TestFixture::real_type> c = { 2.5, 4.5, 6.5, 8.5, 10.5 };
    EXPECT_EQ(this->a + this->b, c);
    EXPECT_EQ(this->b + this->a, c);
}
TYPED_TEST(Operators, operator_add_compound) {
    // compound addition using two vectors
    const std::vector<typename TestFixture::real_type> c = { 2.5, 4.5, 6.5, 8.5, 10.5 };
    EXPECT_EQ(this->a += this->b, c);
}
TYPED_TEST(Operators, operator_add_scalar_binary) {
    // binary addition using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { 2.5, 3.5, 4.5, 5.5, 6.5 };
    EXPECT_EQ(this->a + this->scalar, c);
    EXPECT_EQ(this->scalar + this->a, c);
}
TYPED_TEST(Operators, operator_add_scalar_compound) {
    // compound addition using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { 2.5, 3.5, 4.5, 5.5, 6.5 };
    EXPECT_EQ(this->a += this->scalar, c);
}
TYPED_TEST(Operators, operator_add_binary_empty) {
    // binary addition using two empty vectors
    EXPECT_EQ(this->empty + this->empty, this->empty);
}
TYPED_TEST(Operators, operator_add_compound_empty) {
    // compound addition using two empty vectors
    EXPECT_EQ(this->empty += this->empty, this->empty);
}
TYPED_TEST(Operators, operator_add_scalar_binary_empty) {
    // binary addition using an empty vector and a scalar
    EXPECT_EQ(this->empty + this->scalar, this->empty);
    EXPECT_EQ(this->scalar + this->empty, this->empty);
}
TYPED_TEST(Operators, operator_add_scalar_compound_empty) {
    // compound addition using an empty vector and a scalar
    EXPECT_EQ(this->empty += this->scalar, this->empty);
}
TYPED_TEST(OperatorsDeathTest, operator_add_binary) {
    // try to binary add vectors with different sizes
    EXPECT_DEATH(auto ret = this->a + this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(auto ret = this->b + this->a, "Sizes mismatch!: 2 != 4");
}
TYPED_TEST(OperatorsDeathTest, operator_add_compound) {
    // try to compound add vectors with different sizes
    EXPECT_DEATH(this->a += this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->b += this->a, "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_subtract_binary) {
    // binary subtraction using two vectors
    {
        const std::vector<typename TestFixture::real_type> c = { -0.5, -0.5, -0.5, -0.5, -0.5 };
        EXPECT_EQ(this->a - this->b, c);
    }
    {
        const std::vector<typename TestFixture::real_type> c = { 0.5, 0.5, 0.5, 0.5, 0.5 };
        EXPECT_EQ(this->b - this->a, c);
    }
}
TYPED_TEST(Operators, operator_subtract_compound) {
    // compound subtraction using two vectors
    const std::vector<typename TestFixture::real_type> c = { -0.5, -0.5, -0.5, -0.5, -0.5 };
    EXPECT_EQ(this->a -= this->b, c);
}
TYPED_TEST(Operators, operator_subtract_scalar_binary) {
    // binary subtraction using a vector and a scalar
    {
        const std::vector<typename TestFixture::real_type> c = { -0.5, 0.5, 1.5, 2.5, 3.5 };
        EXPECT_EQ(this->a - this->scalar, c);
    }
    {
        const std::vector<typename TestFixture::real_type> c = { 0.5, -0.5, -1.5, -2.5, -3.5 };
        EXPECT_EQ(this->scalar - this->a, c);
    }
}
TYPED_TEST(Operators, operator_subtract_scalar_compound) {
    // compound subtraction using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { -0.5, 0.5, 1.5, 2.5, 3.5 };
    EXPECT_EQ(this->a -= this->scalar, c);
}
TYPED_TEST(Operators, operator_subtract_binary_empty) {
    // binary subtraction using two empty vectors
    EXPECT_EQ(this->empty - this->empty, this->empty);
}
TYPED_TEST(Operators, operator_subtract_compound_empty) {
    // compound subtraction using two empty vectors
    EXPECT_EQ(this->empty -= this->empty, this->empty);
}
TYPED_TEST(Operators, operator_subtract_scalar_binary_empty) {
    // binary subtraction using an empty vector and a scalar
    EXPECT_EQ(this->empty - this->scalar, this->empty);
    EXPECT_EQ(this->scalar - this->empty, this->empty);
}
TYPED_TEST(Operators, operator_subtract_scalar_compound_empty) {
    // compound subtraction using an empty vector and a scalar
    EXPECT_EQ(this->empty -= this->scalar, this->empty);
}
TYPED_TEST(OperatorsDeathTest, operator_subtract_binary) {
    // try to binary subtract vectors with different sizes
    EXPECT_DEATH(auto ret = this->a - this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(auto ret = this->b - this->a, "Sizes mismatch!: 2 != 4");
}
TYPED_TEST(OperatorsDeathTest, operator_subtract_compound) {
    // try to compound subtract vectors with different sizes
    EXPECT_DEATH(this->a -= this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->b -= this->a, "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_multiply_binary) {
    // binary multiplication using two vectors
    const std::vector<typename TestFixture::real_type> c = { 1.5, 5, 10.5, 18, 27.5 };
    EXPECT_EQ(this->a * this->b, c);
    EXPECT_EQ(this->b * this->a, c);
}
TYPED_TEST(Operators, operator_multiply_compound) {
    // compound multiplication using two vectors
    const std::vector<typename TestFixture::real_type> c = { 1.5, 5, 10.5, 18, 27.5 };
    EXPECT_EQ(this->a *= this->b, c);
}
TYPED_TEST(Operators, operator_multiply_scalar_binary) {
    // binary multiplication using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { 1.5, 3, 4.5, 6, 7.5 };
    EXPECT_EQ(this->a * this->scalar, c);
    EXPECT_EQ(this->scalar * this->a, c);
}
TYPED_TEST(Operators, operator_multiply_scalar_compound) {
    // compound multiplication using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { 1.5, 3, 4.5, 6, 7.5 };
    EXPECT_EQ(this->a *= this->scalar, c);
}
TYPED_TEST(Operators, operator_multiply_binary_empty) {
    // binary multiplication using two empty vectors
    EXPECT_EQ(this->empty * this->empty, this->empty);
}
TYPED_TEST(Operators, operator_multiply_compound_empty) {
    // compound multiplication using two empty vectors
    EXPECT_EQ(this->empty *= this->empty, this->empty);
}
TYPED_TEST(Operators, operator_multiply_scalar_binary_empty) {
    // binary multiplication using an empty vector and a scalar
    EXPECT_EQ(this->empty * this->scalar, this->empty);
    EXPECT_EQ(this->scalar * this->empty, this->empty);
}
TYPED_TEST(Operators, operator_multiply_scalar_compound_empty) {
    // compound multiplication using an empty vector and a scalar
    EXPECT_EQ(this->empty *= this->scalar, this->empty);
}
TYPED_TEST(OperatorsDeathTest, operator_multiply_binary) {
    // try to binary multiply vectors with different sizes
    EXPECT_DEATH(auto ret = this->a * this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(auto ret = this->b * this->a, "Sizes mismatch!: 2 != 4");
}
TYPED_TEST(OperatorsDeathTest, operator_multiply_compound) {
    // try to compound multiply vectors with different sizes
    EXPECT_DEATH(this->a *= this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->b *= this->a, "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_divide_binary) {
    // binary division using two vectors
    {
        const std::vector<typename TestFixture::real_type> c = { 1.0 / 1.5, 2.0 / 2.5, 3.0 / 3.5, 4.0 / 4.5, 5.0 / 5.5 };
        EXPECT_EQ(this->a / this->b, c);
    }
    {
        const std::vector<typename TestFixture::real_type> c = { 1.5, 2.5 / 2.0, 3.5 / 3.0, 4.5 / 4.0, 5.5 / 5.0 };
        EXPECT_EQ(this->b / this->a, c);
    }
}
TYPED_TEST(Operators, operator_divide_compound) {
    // compound division using two vectors
    const std::vector<typename TestFixture::real_type> c = { 1.0 / 1.5, 2.0 / 2.5, 3.0 / 3.5, 4.0 / 4.5, 5.0 / 5.5 };
    EXPECT_EQ(this->a /= this->b, c);
}
TYPED_TEST(Operators, operator_divide_scalar_binary) {
    // binary division using a vector and a scalar
    {
        const std::vector<typename TestFixture::real_type> c = { 1.0 / 1.5, 2.0 / 1.5, 2.0, 4.0 / 1.5, 5.0 / 1.5 };
        EXPECT_EQ(this->a / this->scalar, c);
    }
    {
        const std::vector<typename TestFixture::real_type> c = { 1.5, 1.5 / 2.0, 0.5, 1.5 / 4.0, 1.5 / 5.0 };
        EXPECT_EQ(this->scalar / this->a, c);
    }
}
TYPED_TEST(Operators, operator_divide_scalar_compound) {
    // compound division using a vector and a scalar
    const std::vector<typename TestFixture::real_type> c = { 1.0 / 1.5, 2.0 / 1.5, 2.0, 4.0 / 1.5, 5.0 / 1.5 };
    EXPECT_EQ(this->a /= this->scalar, c);
}
TYPED_TEST(Operators, operator_divide_binary_empty) {
    // binary division using two empty vectors
    EXPECT_EQ(this->empty / this->empty, this->empty);
}
TYPED_TEST(Operators, operator_divide_compound_empty) {
    // compound division using two empty vectors
    EXPECT_EQ(this->empty /= this->empty, this->empty);
}
TYPED_TEST(Operators, operator_divide_scalar_binary_empty) {
    // binary division using an empty vector and a scalar
    EXPECT_EQ(this->empty / this->scalar, this->empty);
    EXPECT_EQ(this->scalar / this->empty, this->empty);
}
TYPED_TEST(Operators, operator_divide_scalar_compound_empty) {
    // compound division using an empty vector and a scalar
    EXPECT_EQ(this->empty /= this->scalar, this->empty);
}
TYPED_TEST(OperatorsDeathTest, operator_divide_binary) {
    // try to binary division vectors with different sizes
    EXPECT_DEATH(auto ret = this->a / this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(auto ret = this->b / this->a, "Sizes mismatch!: 2 != 4");
}
TYPED_TEST(OperatorsDeathTest, operator_divide_compound) {
    // try to compound division vectors with different sizes
    EXPECT_DEATH(this->a /= this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(this->b /= this->a, "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_dot_function) {
    // calculate dot product using the dot function
    EXPECT_EQ(dot(this->a, this->b), 62.5);
    EXPECT_EQ(dot(this->b, this->a), 62.5);
}
TYPED_TEST(Operators, operator_dot_transposed) {
    // calculate dot product using the transposed overload function
    EXPECT_EQ(transposed{ this->a } * this->b, 62.5);
    EXPECT_EQ(transposed{ this->b } * this->a, 62.5);
}
TYPED_TEST(OperatorsDeathTest, operator_dot_function) {
    // try to calculate the dot product with vectors of different sizes
    EXPECT_DEATH(std::ignore = dot(this->a, this->b), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = dot(this->b, this->a), "Sizes mismatch!: 2 != 4");
}
TYPED_TEST(OperatorsDeathTest, operator_dot_transposed) {
    // try to calculate the dot product with vectors of different sizes
    EXPECT_DEATH(std::ignore = transposed{ this->a } * this->b, "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = transposed{ this->b } * this->a, "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_sum) {
    // sum vector elements
    EXPECT_EQ(sum(this->a), 15);
    EXPECT_EQ(sum(this->b), 17.5);
}

TYPED_TEST(Operators, operator_squared_euclidean_dist) {
    // calculate the squared euclidean distance between two vectors
    EXPECT_EQ(squared_euclidean_dist(this->a, this->b), 1.25);
}
TYPED_TEST(OperatorsDeathTest, operator_squared_euclidean_dist) {
    // try to calculate the squared euclidean distance between two vectors with different distance
    EXPECT_DEATH(std::ignore = squared_euclidean_dist(this->a, this->b), "Sizes mismatch!: 4 != 2");
    EXPECT_DEATH(std::ignore = squared_euclidean_dist(this->b, this->a), "Sizes mismatch!: 2 != 4");
}

TYPED_TEST(Operators, operator_sign_positive) {
    EXPECT_EQ(sign(typename TestFixture::real_type{ 1.6 }), 1);
    EXPECT_EQ(sign(typename TestFixture::real_type{ 3 }), 1);
}
TYPED_TEST(Operators, operator_sign_negative) {
    EXPECT_EQ(sign(typename TestFixture::real_type{ -2.4 }), -1);
    EXPECT_EQ(sign(typename TestFixture::real_type{ -4 }), -1);
    EXPECT_EQ(sign(typename TestFixture::real_type{ 0 }), -1);
}
