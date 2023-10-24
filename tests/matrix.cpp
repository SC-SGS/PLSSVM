/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Test the plssvm::matrix class and related functions like the plssvm::layout_type.
 */

#include "plssvm/matrix.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::matrix_exception

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{real_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::generate_random_matrix

#include "gtest/gtest-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ASSERT_EQ, SCOPED_TRACE, FAIL,
                                   // ::testing::{Test, StaticAssertTypeEq}

#include <algorithm>  // std::swap
#include <array>      // std::array
#include <cstddef>    // std::size_t
#include <sstream>    // std::istringstream, std::ostringstream
#include <string>     // std::string
#include <tuple>      // std::ignore
#include <vector>     // std::vector

// check whether the plssvm::layout_type -> std::string conversions are correct
TEST(LayoutType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::layout_type::aos, "aos");
    EXPECT_CONVERSION_TO_STRING(plssvm::layout_type::soa, "soa");
}
TEST(LayoutType, to_string_unknown) {
    // check conversions to std::string from unknown layout_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::layout_type>(2), "unknown");
}

// check whether the std::string -> plssvm::layout_type conversions are correct
TEST(LayoutType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("aos", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("AoS", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("Array-of-Structs", plssvm::layout_type::aos);
    EXPECT_CONVERSION_FROM_STRING("soa", plssvm::layout_type::soa);
    EXPECT_CONVERSION_FROM_STRING("SoA", plssvm::layout_type::soa);
    EXPECT_CONVERSION_FROM_STRING("Struct-of-Arrays", plssvm::layout_type::soa);
}
TEST(LayoutType, from_string_unknown) {
    // foo isn't a valid layout_type
    std::istringstream input{ "foo" };
    plssvm::layout_type layout{};
    input >> layout;
    EXPECT_TRUE(input.fail());
}

TEST(LayoutType, layout_type_to_full_string) {
    // check conversion from plssvm::classification_type to a full string
    EXPECT_EQ(plssvm::layout_type_to_full_string(plssvm::layout_type::aos), "Array-of-Structs");
    EXPECT_EQ(plssvm::layout_type_to_full_string(plssvm::layout_type::soa), "Struct-of-Arrays");
}
TEST(LayoutType, layout_type_to_full_string_unknown) {
    // check conversion from unknown classification_typ to a full string
    EXPECT_EQ(plssvm::layout_type_to_full_string(static_cast<plssvm::layout_type>(2)), "unknown");
}

template <typename T>
class Matrix : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};
TYPED_TEST_SUITE(Matrix, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(Matrix, construct_default) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // default construct matrix
    const plssvm::matrix<real_type, layout> matr{};

    // check content
    EXPECT_EQ(matr.num_rows(), 0);
    EXPECT_EQ(matr.num_cols(), 0);
    EXPECT_TRUE(matr.empty());
}

TYPED_TEST(Matrix, construct_from_same_matrix) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(3, 2);

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr };

    // both matrices should be identical
    EXPECT_EQ(matr, new_matr);
}
TYPED_TEST(Matrix, construct_from_other_matrix) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix with the opposite layout type
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout == plssvm::layout_type::aos ? plssvm::layout_type::soa : plssvm::layout_type::aos>>(3, 2);

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr };

    // both matrices should be identical
    EXPECT_EQ(new_matr.layout(), layout);
    ASSERT_EQ(matr.num_rows(), new_matr.num_rows());
    ASSERT_EQ(matr.num_cols(), new_matr.num_cols());
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr(row, col), new_matr(row, col));
        }
    }
}

TYPED_TEST(Matrix, construct_with_size) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ 3, 2 };

    // check content
    EXPECT_EQ(matr.num_rows(), 3);
    EXPECT_EQ(matr.num_cols(), 2);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.num_entries(), [](const real_type val) { return val == real_type{}; }));
}
TYPED_TEST(Matrix, construct_with_size_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ 0, 0 };

    // check content
    EXPECT_EQ(matr.num_rows(), 0);
    EXPECT_EQ(matr.num_cols(), 0);
}
TYPED_TEST(Matrix, construct_with_size_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 0, 2 }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}
TYPED_TEST(Matrix, construct_with_size_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 2, 0 }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_default_value) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ 2, 3, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.num_rows(), 2);
    EXPECT_EQ(matr.num_cols(), 3);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.num_entries(), [](const real_type val) { return val == real_type{ 3.1415 }; }));
}
TYPED_TEST(Matrix, construct_with_size_and_default_value_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ 0, 0, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.num_rows(), 0);
    EXPECT_EQ(matr.num_cols(), 0);
}
TYPED_TEST(Matrix, construct_with_size_and_default_value_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 0, 2, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}
TYPED_TEST(Matrix, construct_with_size_and_default_value_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 2, 0, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_vector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 };
    const plssvm::matrix<real_type, layout> matr{ 2, 3, data };

    // check content
    EXPECT_EQ(matr.num_rows(), 2);
    EXPECT_EQ(matr.num_cols(), 3);
    ASSERT_EQ(matr.num_entries(), 6);
    for (std::size_t i = 0; i < matr.num_entries(); ++i) {
        EXPECT_FLOATING_POINT_EQ(*(matr.data() + i), data[i]);
    }
}
TYPED_TEST(Matrix, construct_with_size_and_vector_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ 0, 0, std::vector<real_type>{} };

    // check content
    EXPECT_EQ(matr.num_rows(), 0);
    EXPECT_EQ(matr.num_cols(), 0);
}
TYPED_TEST(Matrix, construct_with_size_and_vector_value_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 0, 2, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}
TYPED_TEST(Matrix, construct_with_size_and_vector_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ 2, 0, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_from_2D_vector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.num_rows(), matr_2D.size());
    ASSERT_EQ(matr.num_cols(), matr_2D.front().size());
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);
}
TYPED_TEST(Matrix, construct_from_2D_vector_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ { } };

    // check content
    ASSERT_EQ(matr.num_rows(), 0);
    ASSERT_EQ(matr.num_cols(), 0);
}
TYPED_TEST(Matrix, construct_from_2D_vector_invalid_columns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with mismatching column sizes
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ { { real_type{ 0.0 }, real_type{ 0.0 } }, { real_type{ 0.0 } } } }),
                      plssvm::matrix_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}
TYPED_TEST(Matrix, construct_from_2D_vector_empty_columns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with empty columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ { {}, {} } }),
                      plssvm::matrix_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(Matrix, shape) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 42, 4 };

    // check getter
    EXPECT_EQ(matr.shape(), (std::array<std::size_t, 2>{ 42, 4 }));
}
TYPED_TEST(Matrix, num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 42, 4 };

    // check getter
    EXPECT_EQ(matr.num_rows(), 42);
}
TYPED_TEST(Matrix, num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 42, 4 };

    // check getter
    EXPECT_EQ(matr.num_cols(), 4);
}
TYPED_TEST(Matrix, num_entries) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 42, 4 };

    // check getter
    EXPECT_EQ(matr.num_entries(), 168);
}
TYPED_TEST(Matrix, empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 42, 4 };

    // check getter
    EXPECT_FALSE(matr.empty());
    EXPECT_TRUE((plssvm::matrix<real_type, layout>{}).empty());
}

TYPED_TEST(Matrix, layout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 4, 4 };

    // check getter
    EXPECT_EQ(matr.layout(), layout);
}

TYPED_TEST(Matrix, function_call_operator) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.num_rows(), matr_2D.size());
    ASSERT_EQ(matr.num_cols(), matr_2D.front().size());
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr(row, col), matr_2D[row][col]);
        }
    }
}
TYPED_TEST(Matrix, function_call_operator_const) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.num_rows(), matr_2D.size());
    ASSERT_EQ(matr.num_cols(), matr_2D.front().size());
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr(row, col), matr_2D[row][col]);
        }
    }
}
TYPED_TEST(Matrix, at) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.num_rows(), matr_2D.size());
    ASSERT_EQ(matr.num_cols(), matr_2D.front().size());
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr.at(row, col), matr_2D[row][col]);
        }
    }
}
TYPED_TEST(Matrix, at_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ 2, 2 };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}
TYPED_TEST(Matrix, at_const) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.num_rows(), matr_2D.size());
    ASSERT_EQ(matr.num_cols(), matr_2D.front().size());
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr.at(row, col), matr_2D[row][col]);
        }
    }
}
TYPED_TEST(Matrix, at_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 2, 2 };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}

TYPED_TEST(Matrix, to_2D_vector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 1.1 }, real_type{ 1.2 } } };

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);
}

TYPED_TEST(Matrix, swap_member_function) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data
    const std::vector<std::vector<real_type>> matr1_2D = { { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } }, { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } } };
    const std::vector<std::vector<real_type>> matr2_2D = { { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 0.3 }, real_type{ 0.4 } }, { real_type{ 0.5 }, real_type{ 0.6 } } };

    // create two matrices and swap their content
    plssvm::matrix<real_type, layout> matr1{ matr1_2D };
    plssvm::matrix<real_type, layout> matr2{ matr2_2D };

    // swap both matrices
    matr1.swap(matr2);

    // check the content of matr1
    ASSERT_EQ(matr1.num_rows(), 3);
    ASSERT_EQ(matr1.num_cols(), 2);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.num_rows(), 2);
    ASSERT_EQ(matr2.num_cols(), 3);
    EXPECT_EQ(matr2.to_2D_vector(), matr1_2D);
}
TYPED_TEST(Matrix, swap_free_function) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data
    const std::vector<std::vector<real_type>> matr1_2D = { { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } }, { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } } };
    const std::vector<std::vector<real_type>> matr2_2D = { { real_type{ 0.1 }, real_type{ 0.2 } }, { real_type{ 0.3 }, real_type{ 0.4 } }, { real_type{ 0.5 }, real_type{ 0.6 } } };

    // create two matrices and swap their content
    plssvm::matrix<real_type, layout> matr1{ matr1_2D };
    plssvm::matrix<real_type, layout> matr2{ matr2_2D };

    // swap both matrices
    using std::swap;
    swap(matr1, matr2);

    // check the content of matr1
    ASSERT_EQ(matr1.num_rows(), 3);
    ASSERT_EQ(matr1.num_cols(), 2);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.num_rows(), 2);
    ASSERT_EQ(matr2.num_cols(), 3);
    EXPECT_EQ(matr2.to_2D_vector(), matr1_2D);
}

TYPED_TEST(Matrix, operator_equal) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create matrices
    const plssvm::matrix<real_type, layout> matr1{ 3, 2 };
    const plssvm::matrix<real_type, layout> matr2{ 2, 3 };
    const plssvm::matrix<real_type, layout> matr3{ 3, 3 };
    const plssvm::matrix<real_type, layout> matr4{ 3, 3, real_type{ 3.1415 } };
    const plssvm::matrix<real_type, layout> matr5{ 3, 3, real_type{} };

    // check for equality
    EXPECT_FALSE(matr1 == matr2);
    EXPECT_FALSE(matr1 == matr3);
    EXPECT_FALSE(matr2 == matr3);
    EXPECT_FALSE(matr3 == matr4);
    EXPECT_TRUE(matr3 == matr5);
    EXPECT_TRUE(matr4 == matr4);
}
TYPED_TEST(Matrix, operator_unequal) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create matrices
    const plssvm::matrix<real_type, layout> matr1{ 3, 2 };
    const plssvm::matrix<real_type, layout> matr2{ 2, 3 };
    const plssvm::matrix<real_type, layout> matr3{ 3, 3 };
    const plssvm::matrix<real_type, layout> matr4{ 3, 3, real_type{ 3.1415 } };
    const plssvm::matrix<real_type, layout> matr5{ 3, 3, real_type{} };

    // check for equality
    EXPECT_TRUE(matr1 != matr2);
    EXPECT_TRUE(matr1 != matr3);
    EXPECT_TRUE(matr2 != matr3);
    EXPECT_TRUE(matr3 != matr4);
    EXPECT_FALSE(matr3 != matr5);
    EXPECT_FALSE(matr4 != matr4);
}

TYPED_TEST(Matrix, output_operator) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data
    const std::vector<std::vector<real_type>> matr_2D = { { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } }, { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } } };

    // create matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check
    std::string correct_output{};
    for (const std::vector<real_type> &row : matr_2D) {
        correct_output += fmt::format("{:.10e} \n", fmt::join(row, " "));
    }
    correct_output.pop_back();  // remove last newline
    EXPECT_CONVERSION_TO_STRING(matr, correct_output);
}

TYPED_TEST(Matrix, matrix_shorthands) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    if constexpr (layout == plssvm::layout_type::aos) {
        ::testing::StaticAssertTypeEq<plssvm::matrix<real_type, layout>, plssvm::aos_matrix<real_type>>();
    } else if constexpr (layout == plssvm::layout_type::soa) {
        ::testing::StaticAssertTypeEq<plssvm::matrix<real_type, layout>, plssvm::soa_matrix<real_type>>();
    }else {
        FAIL() << "Unrecognized layout type!";
    }
}

template <typename T>
class MatrixDeathTest : public Matrix<T> {};
TYPED_TEST_SUITE(MatrixDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixDeathTest, function_call_operator_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ 2, 2 };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}
TYPED_TEST(MatrixDeathTest, function_call_operator_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ 2, 2 };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}