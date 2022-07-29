/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the std::vector arithmetic operator overloads.
 */

#include "plssvm/detail/operators.hpp"  // plssvm::operators::{*}

#include "gtest/gtest.h"  // ::testing::Types, ::testing::Test, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_DEATH

#include <vector>  // std::vector

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

// testsuite for "normal" tests
template <typename T>
class BaseOperators : public ::testing::Test {};
TYPED_TEST_SUITE(BaseOperators, floating_point_types);

// testsuite for death tests
template <typename T>
class BaseOperatorsDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(BaseOperatorsDeathTest, floating_point_types);

TYPED_TEST(BaseOperators, operator_add) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    std::vector<real_type> b = { 5, 6, 7, 8, 9 };
    const real_type scalar = 42.5;

    // addition using two vectors
    std::vector<real_type> c = { 5, 7, 9, 11, 13 };
    EXPECT_EQ(a + b, c);
    EXPECT_EQ(b + a, c);
    EXPECT_EQ(b += a, c);

    // addition using a vector and a scalar
    c = { 42.5, 43.5, 44.5, 45.5, 46.5 };
    EXPECT_EQ(a + scalar, c);
    EXPECT_EQ(scalar + a, c);
    EXPECT_EQ(a += scalar, c);

    // add empty vectors
    const std::vector<real_type> empty_vec;
    a.clear();
    ASSERT_EQ(a.size(), 0);

    EXPECT_EQ(a + a, empty_vec);
    EXPECT_EQ(a += a, empty_vec);
    EXPECT_EQ(a + scalar, empty_vec);
    EXPECT_EQ(scalar + a, empty_vec);
    EXPECT_EQ(a += scalar, empty_vec);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_add) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] std::vector<real_type> ret;

    // try to add vectors with different sizes
    std::vector<real_type> a = { 0, 1, 2, 3 };
    std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = a + b, "");
    EXPECT_DEATH(ret = b + a, "");
    EXPECT_DEATH(a += b, "");
    EXPECT_DEATH(b += a, "");
}


TYPED_TEST(BaseOperators, operator_subtract) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    std::vector<real_type> b = { 5, 6, 7, 8, 9 };
    const real_type scalar = 1.5;

    // subtraction using two vectors
    std::vector<real_type> c = { -5, -5, -5, -5, -5 };
    EXPECT_EQ(a - b, c);
    c = { 5, 5, 5, 5, 5 };
    EXPECT_EQ(b - a, c);
    EXPECT_EQ(b -= a, c);

    // subtraction using a vector and a scalar
    c = { -1.5, -0.5, 0.5, 1.5, 2.5 };
    EXPECT_EQ(a - scalar, c);
    c = { 1.5, 0.5, -0.5, -1.5, -2.5 };
    EXPECT_EQ(scalar - a, c);
    c = { -1.5, -0.5, 0.5, 1.5, 2.5 };
    EXPECT_EQ(a -= scalar, c);

    // subtract empty vectors
    const std::vector<real_type> empty_vec;
    a.clear();
    ASSERT_EQ(a.size(), 0);

    EXPECT_EQ(a - a, empty_vec);
    EXPECT_EQ(a -= a, empty_vec);
    EXPECT_EQ(a - scalar, empty_vec);
    EXPECT_EQ(scalar - a, empty_vec);
    EXPECT_EQ(a -= scalar, empty_vec);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_subtract) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] std::vector<real_type> ret;

    // try to subtract vectors with different sizes
    std::vector<real_type> a = { 0, 1, 2, 3 };
    std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = a - b, "");
    EXPECT_DEATH(ret = b - a, "");
    EXPECT_DEATH(a -= b, "");
    EXPECT_DEATH(b -= a, "");
}


TYPED_TEST(BaseOperators, operator_multiply) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    std::vector<real_type> b = { 5, 6, 7, 8, 9 };
    const real_type scalar = 1.5;

    // multiplication using two vectors
    std::vector<real_type> c = { 0, 6, 14, 24, 36 };
    EXPECT_EQ(a * b, c);
    EXPECT_EQ(b * a, c);
    EXPECT_EQ(b *= a, c);

    // multiplication using a vector and a scalar
    c = { 0, 1.5, 3, 4.5, 6 };
    EXPECT_EQ(a * scalar, c);
    EXPECT_EQ(scalar * a, c);
    EXPECT_EQ(a *= scalar, c);

    // multiply empty vectors
    const std::vector<real_type> empty_vec;
    a.clear();
    ASSERT_EQ(a.size(), 0);

    EXPECT_EQ(a * a, empty_vec);
    EXPECT_EQ(a *= a, empty_vec);
    EXPECT_EQ(a * scalar, empty_vec);
    EXPECT_EQ(scalar * a, empty_vec);
    EXPECT_EQ(a *= scalar, empty_vec);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_multiply) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] std::vector<real_type> ret;

    // try to multiply vectors with different sizes
    std::vector<real_type> a = { 0, 1, 2, 3 };
    std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = a * b, "");
    EXPECT_DEATH(ret = b * a, "");
    EXPECT_DEATH(a *= b, "");
    EXPECT_DEATH(b *= a, "");
}


TYPED_TEST(BaseOperators, operator_divide) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    std::vector<real_type> a = { 1, 1, 2, 3, 4 };
    std::vector<real_type> b = { 5, 6, 7, 8, 9 };
    const real_type scalar = 1.5;

    // division using two vectors
    std::vector<real_type> c = { 1. / 5., 1. / 6., 2. / 7., 3. / 8., 4. / 9. };
    EXPECT_EQ(a / b, c);
    c = { 5, 6, 3.5, 8. / 3., 2.25 };
    EXPECT_EQ(b / a, c);
    EXPECT_EQ(b /= a, c);

    // division using a vector and a scalar
    c = { 1. / 1.5, 1. / 1.5, 2. / 1.5, 2, 4. / 1.5 };
    EXPECT_EQ(a / scalar, c);
    c = { 1.5, 1.5, 0.75, 0.5, 0.375 };
    EXPECT_EQ(scalar / a, c);
    c = { 1. / 1.5, 1. / 1.5, 2. / 1.5, 2, 4. / 1.5 };
    EXPECT_EQ(a /= scalar, c);

    // divide empty vectors
    const std::vector<real_type> empty_vec;
    a.clear();
    ASSERT_EQ(a.size(), 0);

    EXPECT_EQ(a / a, empty_vec);
    EXPECT_EQ(a /= a, empty_vec);
    EXPECT_EQ(a / scalar, empty_vec);
    EXPECT_EQ(scalar / a, empty_vec);
    EXPECT_EQ(a /= scalar, empty_vec);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_divide) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] std::vector<real_type> ret;

    // try to divide vectors with different sizes
    std::vector<real_type> a = { 0, 1, 2, 3 };
    std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = a / b, "");
    EXPECT_DEATH(ret = b / a, "");
    EXPECT_DEATH(a /= b, "");
    EXPECT_DEATH(b /= a, "");
}


TYPED_TEST(BaseOperators, operator_dot) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    const std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    const std::vector<real_type> b = { 5, 6, 7, 8, 9 };

    // calculate dot product
    EXPECT_EQ(dot(a, b), 80);
    EXPECT_EQ(dot(b, a), 80);
    EXPECT_EQ(transposed{ a } * b, 80);
    EXPECT_EQ(transposed{ b } * a, 80);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_dot) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] real_type ret;

    // try to calculate the dot product with vectors of different sizes
    std::vector<real_type> a = { 0, 1, 2, 3 };
    const std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = dot(a, b), "");
    EXPECT_DEATH(ret = dot(b, a), "");
    EXPECT_DEATH(ret = transposed{ a } * b, "");
    EXPECT_DEATH(ret = transposed{ b } * a, "");
}


TYPED_TEST(BaseOperators, operator_sum) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    // sum vector elements
    const std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    EXPECT_EQ(sum(a), 10);

    const std::vector<real_type> b = { -1.5, 4, -5.5, -3.5, 1.5 };
    EXPECT_EQ(sum(b), -5);
}


TYPED_TEST(BaseOperators, operator_squared_euclidean_dist) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    const std::vector<real_type> a = { 0, 1, 2, 3, 4 };
    const std::vector<real_type> b = { 5, 6, 7, 8, 9 };

    // calculate the squared euclidean distance between two vectors
    EXPECT_EQ(squared_euclidean_dist(a, b), 125);
}

TYPED_TEST(BaseOperatorsDeathTest, operator_squared_euclidean_dist) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    [[maybe_unused]] real_type ret;

    // try to calculate the squared euclidean distance between two vectors with different distance
    std::vector<real_type> a = { 0, 1, 2, 3 };
    const std::vector<real_type> b = { 4, 5 };
    EXPECT_DEATH(ret = squared_euclidean_dist(a, b), "");
    EXPECT_DEATH(ret = squared_euclidean_dist(b, a), "");
}


TYPED_TEST(BaseOperators, operator_sign) {
    using real_type = TypeParam;
    using namespace plssvm::operators;

    EXPECT_EQ(sign(real_type{ -2.4 }), -1);
    EXPECT_EQ(sign(real_type{ -4 }), -1);
    EXPECT_EQ(sign(real_type{ 0 }), -1);
    EXPECT_EQ(sign(real_type{ 1.6 }), 1);
    EXPECT_EQ(sign(real_type{ 3 }), 1);
}
