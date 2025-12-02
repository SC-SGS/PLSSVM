/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Test the plssvm::matrix class operation overloads and math functions.
 */

#include "plssvm/matrix.hpp"

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_EQ
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{real_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}

#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_DEATH, ::testing::Test

#include <tuple>   // std::ignore
#include <vector>  // std::vector

template <typename T>
class MatrixOperations : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        A_ = plssvm::matrix<fixture_real_type, fixture_layout>{ { { 1, 2, 3 }, { 4, 5, 6 } } };
        B_ = plssvm::matrix<fixture_real_type, fixture_layout>{ { { 1.5, 2.5, 3.5 }, { 4.5, 5.5, 6.5 } } };
        c_ = std::vector<fixture_real_type>{ 1.5, -2.5 };
        scalar_ = fixture_real_type{ 1.5 };
    }

    /**
     * @brief Return the sample matrix @p A.
     * @return the matrix vector (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_A() noexcept { return A_; }

    /**
     * @brief Return the sample matrix @p B.
     * @return the sample matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_B() noexcept { return B_; }

    /**
     * @brief Return the sample vector @p c.
     * @return the sample vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<fixture_real_type> &get_c() noexcept { return c_; }

    /**
     * @brief Return the empty matrix.
     * @return the empty matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_empty() noexcept { return empty_; }

    /**
     * @brief Return the sample scalar.
     * @return the scalar (`[[nodiscard]]`)
     */
    [[nodiscard]] fixture_real_type &get_scalar() noexcept { return scalar_; }

  private:
    /// Sample matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> A_{};
    /// Sample matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> B_{};
    /// Sample vector to test the different operations.
    std::vector<fixture_real_type> c_{};
    /// Empty matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> empty_{};
    /// Sample scalar to test the different operations.
    fixture_real_type scalar_{};
};

TYPED_TEST_SUITE(MatrixOperations, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

template <typename T>
class MatrixOperationsDeathTest : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        A_ = plssvm::matrix<fixture_real_type, fixture_layout>{ { { 1, 2, 3 }, { 4, 5, 6 } } };
        B_ = plssvm::matrix<fixture_real_type, fixture_layout>{ { { 1.5, 2.5 }, { 3.5, 4.5 }, { 5.5, 6.5 } } };
    }

    /**
     * @brief Return the sample matrix @p A.
     * @return the matrix vector (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_A() noexcept { return A_; }

    /**
     * @brief Return the sample matrix @p B.
     * @return the sample matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_B() noexcept { return B_; }

    /**
     * @brief Return the empty matrix.
     * @return the empty matrix (`[[nodiscard]]`)
     */
    [[nodiscard]] plssvm::matrix<fixture_real_type, fixture_layout> &get_empty() noexcept { return empty_; }

  private:
    /// Sample matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> A_{};
    /// Sample matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> B_{};
    /// Empty matrix to test the different operations.
    plssvm::matrix<fixture_real_type, fixture_layout> empty_{};
};

TYPED_TEST_SUITE(MatrixOperationsDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixOperations, OperatorScaleBinary) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // scale a matrix using a scalar
    {
        const plssvm::matrix<real_type, layout> C{ { { 1.5, 3.0, 4.5 }, { 6.0, 7.5, 9.0 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_scalar() * this->get_A(), C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 2.25, 3.75, 5.25 }, { 6.75, 8.25, 9.75 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_B() * this->get_scalar(), C);
    }
}

TYPED_TEST(MatrixOperations, OperatorScaleBinaryEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() * this->get_scalar(), this->get_empty());
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_scalar() * this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, OperatorScaleCompound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // scale a matrix using a scalar
    const plssvm::matrix<real_type, layout> C{ { { 1.5, 3.0, 4.5 }, { 6.0, 7.5, 9.0 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() *= this->get_scalar(), C);
}

TYPED_TEST(MatrixOperations, OperatorScaleCompoundEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() *= this->get_scalar(), this->get_empty());
}

TYPED_TEST(MatrixOperations, OperatorAddBinary) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { 2.5, 4.5, 6.5 }, { 8.5, 10.5, 12.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() + this->get_B(), C);
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_B() + this->get_A(), C);
}

TYPED_TEST(MatrixOperations, OperatorAddBinaryEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() + this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, OperatorAddCompound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { 2.5, 4.5, 6.5 }, { 8.5, 10.5, 12.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() += this->get_B(), C);
}

TYPED_TEST(MatrixOperations, OperatorAddCompoundEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() += this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorAddBinary) {
    // try adding two matrices with mismatching sizes
    EXPECT_DEATH(std::ignore = this->get_A() + this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(std::ignore = this->get_B() + this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorAddCompound) {
    // try adding two matrices with mismatching sizes
    EXPECT_DEATH(this->get_A() += this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(this->get_B() += this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperations, OperatorSubtractBinary) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // subtract two matrices from each other
    {
        const plssvm::matrix<real_type, layout> C{ { { -0.5, -0.5, -0.5 }, { -0.5, -0.5, -0.5 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() - this->get_B(), C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 0.5, 0.5, 0.5 }, { 0.5, 0.5, 0.5 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_B() - this->get_A(), C);
    }
}

TYPED_TEST(MatrixOperations, OperatorSubtractBinaryEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() - this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, OperatorSubtractCompound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { -0.5, -0.5, -0.5 }, { -0.5, -0.5, -0.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() -= this->get_B(), C);
}

TYPED_TEST(MatrixOperations, OperatorSubtractCompoundEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() -= this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorSubtractBinary) {
    // try subtracting two matrices with mismatching sizes
    EXPECT_DEATH(std::ignore = this->get_A() - this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(std::ignore = this->get_B() - this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorSubtractCompound) {
    // try subtracting two matrices with mismatching sizes
    EXPECT_DEATH(this->get_A() -= this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(this->get_B() -= this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperations, OperatorMatrixMultiplicationSquare) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    const plssvm::matrix<real_type, layout> square_A{ { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 7.0, 8.0, 9.0 } } };
    const plssvm::matrix<real_type, layout> square_B{ { { 1.5, 2.5, 3.5 }, { 4.5, 5.5, 6.5 }, { 7.5, 8.5, 9.5 } } };

    // matrix-matrix multiplication with squared matrices
    {
        const plssvm::matrix<real_type, layout> C{ { { 33.0, 39.0, 45.0 }, { 73.5, 88.5, 103.5 }, { 114.0, 138.0, 162.0 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(square_A * square_B, C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 36.0, 43.5, 51.0 }, { 72.0, 88.5, 105.0 }, { 108.0, 133.5, 159.0 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(square_B * square_A, C);
    }
}

TYPED_TEST(MatrixOperations, OperatorMatrixMultiplication) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    const plssvm::matrix<real_type, layout> input_A{ { { 1, 2, 3 }, { 4, 5, 6 } } };
    const plssvm::matrix<real_type, layout> input_B{ { { 1.5, 2.5 }, { 4.5, 5.5 }, { 7.5, 8.5 } } };

    // matrix-matrix multiplication
    {
        const plssvm::matrix<real_type, layout> C{ { { 33.0, 39.0 }, { 73.5, 88.5 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(input_A * input_B, C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 11.5, 15.5, 19.5 }, { 26.5, 36.5, 46.5 }, { 41.5, 57.5, 73.5 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(input_B * input_A, C);
    }
}

TYPED_TEST(MatrixOperations, OperatorMatrixMultiplicationEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() * this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorMatrixMultiplication) {
    EXPECT_DEATH(std::ignore = this->get_A() * this->get_empty(), ::testing::HasSubstr("Error: shapes missmatch! (3 (num_cols) != 0 (num_rows))"));
    EXPECT_DEATH(std::ignore = this->get_empty() * this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! (0 (num_cols) != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, OperatorRowwiseDot) {
    using real_type = typename TestFixture::fixture_real_type;

    // rowwise dot product
    const std::vector<real_type> ret{ 17.0, 84.5 };
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_A(), this->get_B()), ret);
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_B(), this->get_A()), ret);
}

TYPED_TEST(MatrixOperations, OperatorRowwiseDotEmpty) {
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_empty(), this->get_empty()), {});
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorRowwiseDot) {
    // sizes missmatch
    EXPECT_DEATH(std::ignore = rowwise_dot({}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! ([0, 0] != [2, 3])"));
    EXPECT_DEATH(std::ignore = rowwise_dot(this->get_A(), this->get_B()), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
}

TYPED_TEST(MatrixOperations, OperatorRowwiseScale) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // rowwise scale
    {
        const plssvm::matrix<real_type, layout> C{ { { 1.5, 3.0, 4.5 }, { -10.0, -12.5, -15.0 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(rowwise_scale(this->get_c(), this->get_A()), C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 2.25, 3.75, 5.25 }, { -11.25, -13.75, -16.25 } } };
        EXPECT_FLOATING_POINT_MATRIX_EQ(rowwise_scale(this->get_c(), this->get_B()), C);
    }
}

TYPED_TEST(MatrixOperations, OperatorRowwiseScaleEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(rowwise_scale({}, this->get_empty()), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorRowwiseScale) {
    using real_type = typename TestFixture::fixture_real_type;

    // sizes missmatch
    EXPECT_DEATH(std::ignore = rowwise_scale({}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = rowwise_scale(std::vector<real_type>{ 1.0, 2.0, 3.0 }, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, OperatorMaskedRowwiseScale) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // rowwise scale
    {
        const plssvm::matrix<real_type, layout> C{ { { 0.0, 0.0, 0.0 }, { -10.0, -12.5, -15.0 } } };
        const std::vector<unsigned long long> mask{ 0, 1 };
        EXPECT_FLOATING_POINT_MATRIX_EQ(masked_rowwise_scale(mask, this->get_c(), this->get_A()), C);
    }
    {
        const plssvm::matrix<real_type, layout> C{ { { 2.25, 3.75, 5.25 }, { 0.0, 0.0, 0.0 } } };
        const std::vector<unsigned long long> mask{ 1, 0 };
        EXPECT_FLOATING_POINT_MATRIX_EQ(masked_rowwise_scale(mask, this->get_c(), this->get_B()), C);
    }
}

TYPED_TEST(MatrixOperations, OperatorMaskedRowwiseScaleEmpty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(masked_rowwise_scale({}, {}, this->get_empty()), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, OperatorMaskedRowwiseScale) {
    using real_type = typename TestFixture::fixture_real_type;

    const std::vector<real_type> scale{ 1.0, 2.0 };
    const std::vector<unsigned long long> mask{ 1, 0 };

    // sizes missmatch
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(mask, {}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(mask, std::vector<real_type>{ 1.0, 2.0, 3.0 }, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(std::vector<unsigned long long>{}, scale, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(std::vector<unsigned long long>{ 1, 0, 1 }, scale, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, OperatorVariance) {
    using real_type = typename TestFixture::fixture_real_type;

    EXPECT_FLOATING_POINT_NEAR(variance(this->get_A()), (real_type{ 17.5 } / static_cast<real_type>(this->get_A().size())));
    EXPECT_FLOATING_POINT_NEAR(variance(this->get_B()), (real_type{ 17.5 } / static_cast<real_type>(this->get_A().size())));
}
