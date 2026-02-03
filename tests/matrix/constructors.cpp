/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Test the plssvm::matrix class constructors.
 */

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::matrix_exception
#include "plssvm/matrix.hpp"
#include "plssvm/shape.hpp"  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{real_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"             // util::{generate_random_matrix, redirect_output}

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, ASSERT_EQ, SCOPED_TRACE, ::testing::Test

#include <cstddef>   // std::size_t
#include <iostream>  // std::clog
#include <vector>    // std::vector

template <typename T>
class MatrixConstructors : public ::testing::Test,
                           public util::redirect_output<&std::clog> {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};

TYPED_TEST_SUITE(MatrixConstructors, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixConstructors, ConstructDefault) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // default construct matrix
    const plssvm::matrix<real_type, layout> matr{};

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSize) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 3, 2 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr.size(), 6);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size(), [](const real_type val) { return val == real_type{}; }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 3, 2 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeZeroNumRows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeZeroNumCols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 3, 2 }, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 3, 2 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 7, 7 }));
    // default values == padding values
    ASSERT_EQ(matr.size_padded(), 49);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{}; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 4, 5 }));

    // default values == padding values
    ASSERT_EQ(matr.size_padded(), 20);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{}; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeEmptyAndZeroPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeZeroNumRowsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeZeroNumColsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValue) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr.size(), 6);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size(), [](const real_type val) { return val == real_type{ 3.1415 }; }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 2, 3 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueZeroNumRows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueZeroNumCols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, real_type{ 3.1415 }, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 6, 8 }));

    // check content while paying attention to padding!
    ASSERT_EQ(matr.size_padded(), 48);
    for (std::size_t row = 0; row < matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < matr.num_rows() && col < matr.num_cols()) {
                EXPECT_EQ(matr(row, col), real_type{ 3.1415 });
            } else {
                EXPECT_EQ(matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, real_type{ 3.1415 }, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 4, 5 }));

    // only padding entries should be present
    ASSERT_EQ(matr.size_padded(), 20);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{ 0.0 }; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueEmptyAndZeroPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, real_type{ 3.1415 }, plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueZeroNumRowsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, real_type{ 3.1415 }, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueZeroNumColsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, real_type{ 3.1415 }, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data = { real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 }, real_type{ 0.4 }, real_type{ 0.5 }, real_type{ 0.6 } };
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, data };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr.size(), 6);
    for (std::size_t i = 0; i < matr.size(); ++i) {
        EXPECT_FLOATING_POINT_EQ(*(matr.data() + i), data[i]);
    }
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 2, 3 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, std::vector<real_type>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorValueZeroNumRows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorZeroNumCols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorSizeMismatch) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 2 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of entries in the matrix (4) must be equal to the size of the data (2)!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data = { real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 }, real_type{ 0.4 }, real_type{ 0.5 }, real_type{ 0.6 } };
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, data, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 6, 8 }));

    // check content while paying attention to padding!
    ASSERT_EQ(matr.size_padded(), 48);
    for (std::size_t row = 0; row < matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < matr.num_rows() && col < matr.num_cols()) {
                const real_type val = layout == plssvm::layout_type::aos ? data[row * 3 + col] : data[col * 2 + row];
                EXPECT_EQ(matr(row, col), val);
            } else {
                EXPECT_EQ(matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, std::vector<real_type>{}, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 4, 5 }));

    // only padding entries should be present
    ASSERT_EQ(matr.size_padded(), 20);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{ 0.0 }; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorEmptyAndZeroPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, std::vector<real_type>{}, plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorValueZeroNumRowsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, std::vector<real_type>(2), plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorZeroNumColsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, std::vector<real_type>(2), plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorSizeMismatchAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 2 }, std::vector<real_type>(3), plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of entries in the matrix (4) must be equal to the size of the data (3)!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointer) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data = { real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 }, real_type{ 0.4 }, real_type{ 0.5 }, real_type{ 0.6 } };
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, data.data() };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr.size(), 6);
    for (std::size_t i = 0; i < matr.size(); ++i) {
        EXPECT_FLOATING_POINT_EQ(*(matr.data() + i), data[i]);
    }
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 2, 3 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data{};
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, data.data() };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndNullptrPointer) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size and a nullptr
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 3 }, nullptr }),
                      plssvm::matrix_exception,
                      "The provided data pointer may not be a nullptr if the matrix size is greater than 0!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerValueZeroNumRows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, data.data() }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerZeroNumCols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, data.data() }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data = { real_type{ 0.1 }, real_type{ 0.2 }, real_type{ 0.3 }, real_type{ 0.4 }, real_type{ 0.5 }, real_type{ 0.6 } };
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, data.data(), plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 6, 8 }));

    // check content while paying attention to padding!
    ASSERT_EQ(matr.size_padded(), 48);
    for (std::size_t row = 0; row < matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < matr.num_rows() && col < matr.num_cols()) {
                const real_type val = layout == plssvm::layout_type::aos ? data[row * 3 + col] : data[col * 2 + row];
                EXPECT_EQ(matr(row, col), val);
            } else {
                EXPECT_EQ(matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data;
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, data.data(), plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 4, 5 }));

    // only padding entries should be present
    ASSERT_EQ(matr.size_padded(), 20);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{ 0.0 }; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndNullptrEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size, padding and a nullptr
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 3 }, nullptr, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The provided data pointer may not be a nullptr if the matrix size is greater than 0!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerEmptyAndZeroPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data;
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, data.data(), plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerValueZeroNumRowsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, data.data(), plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerZeroNumColsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, data.data(), plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(MatrixConstructors, ConstructFromSameMatrixLayout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 3, 2 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr };  // NOLINT(performance-unnecessary-copy-initialization): copy explicitly wanted

    // both matrices should be identical
    EXPECT_EQ(new_matr, matr);
}

TYPED_TEST(MatrixConstructors, ConstructFromOtherMatrixLayout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix with the opposite layout type
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout == plssvm::layout_type::aos ? plssvm::layout_type::soa : plssvm::layout_type::aos>>(plssvm::shape{ 3, 2 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr };

    // both matrices should be identical
    EXPECT_EQ(new_matr.layout(), layout);
    EXPECT_EQ(new_matr.shape(), matr.shape());
    // check padding
    EXPECT_EQ(new_matr.padding(), matr.padding());
    EXPECT_EQ(new_matr.shape_padded(), matr.shape_padded());

    // check content
    ASSERT_EQ(new_matr.size_padded(), matr.size_padded());
    for (std::size_t row = 0; row < new_matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < new_matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < new_matr.num_rows() && col < new_matr.num_cols()) {
                EXPECT_EQ(new_matr(row, col), matr(row, col));
            } else {
                EXPECT_EQ(new_matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructFromSameMatrixLayoutAndSamePadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 3, 2 }, plssvm::shape{ 4, 5 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr, plssvm::shape{ 4, 5 } };

    // both matrices should be identical
    EXPECT_EQ(new_matr, matr);
}

TYPED_TEST(MatrixConstructors, ConstructFromSameMatrixLayoutAndDifferentPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 3, 2 }, plssvm::shape{ 4, 5 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr, plssvm::shape{ 2, 3 } };

    // both matrices shouldn't be identical because of different padding sizes
    EXPECT_NE(new_matr, matr);
    // check content
    EXPECT_EQ(new_matr.layout(), layout);
    EXPECT_EQ(new_matr.shape(), matr.shape());
    // only padding sizes should have changed
    EXPECT_EQ(new_matr.padding(), (plssvm::shape{ 2, 3 }));
    EXPECT_EQ(new_matr.shape_padded(), (plssvm::shape{ 5, 5 }));

    // check content
    ASSERT_EQ(new_matr.size_padded(), 25);
    for (std::size_t row = 0; row < new_matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < new_matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < new_matr.num_rows() && col < new_matr.num_cols()) {
                EXPECT_EQ(new_matr(row, col), matr(row, col));
            } else {
                EXPECT_EQ(new_matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructFromOtherMatrixLayoutAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix with the opposite layout type
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout == plssvm::layout_type::aos ? plssvm::layout_type::soa : plssvm::layout_type::aos>>(plssvm::shape{ 3, 2 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr, plssvm::shape{ 4, 5 } };

    // both matrices shouldn't be identical because of different layout types
    // check content
    EXPECT_EQ(new_matr.layout(), layout);
    EXPECT_EQ(new_matr.shape(), matr.shape());
    // only padding sizes should have changed
    EXPECT_EQ(new_matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(new_matr.shape_padded(), (plssvm::shape{ 7, 7 }));

    // check content
    ASSERT_EQ(new_matr.size_padded(), 49);
    for (std::size_t row = 0; row < new_matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < new_matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < new_matr.num_rows() && col < new_matr.num_cols()) {
                EXPECT_EQ(new_matr(row, col), matr(row, col));
            } else {
                EXPECT_EQ(new_matr(row, col), real_type{ 0.0 });
            }
        }
    }
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 0.3 }, real_type{ 0.4 } } };

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    ASSERT_EQ(matr.size_padded(), 4);
    auto val = static_cast<real_type>(0.1);
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_FLOATING_POINT_EQ(matr(row, col), val);
            val += real_type{ 0.1 };
        }
    }
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);

    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ matr_2D.size(), matr_2D.front().size() }));
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ std::vector<std::vector<real_type>>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 0, 0 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorInvalidColumns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with mismatching column sizes
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ { { real_type{ 0.0 }, real_type{ 0.0 } },
                                                            { real_type{ 0.0 } } } }),
                      plssvm::matrix_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorEmptyColumns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with empty columns
    const std::vector<std::vector<real_type>> matr{ {}, {} };
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ matr }),
                      plssvm::matrix_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 0.3 }, real_type{ 0.4 } } };

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ matr_2D, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    ASSERT_EQ(matr.size_padded(), 42);
    // check content while paying attention to padding!
    auto val = static_cast<real_type>(0.1);
    for (std::size_t row = 0; row < matr.num_rows_padded(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols_padded(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if (row < matr.num_rows() && col < matr.num_cols()) {
                EXPECT_FLOATING_POINT_EQ(matr(row, col), val);
                val += real_type{ 0.1 };
            } else {
                EXPECT_EQ(matr(row, col), real_type{ 0.0 });
            }
        }
    }
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);

    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 6, 7 }));
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorEmptyAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix from a std::vector<std::vector<>>
    const std::vector<std::vector<real_type>> empty{};
    const plssvm::matrix<real_type, layout> matr{ empty, plssvm::shape{ 4, 5 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
    // check padding
    EXPECT_EQ(matr.padding(), (plssvm::shape{ 4, 5 }));
    EXPECT_EQ(matr.shape_padded(), (plssvm::shape{ 4, 5 }));

    // only padding entries should be present
    ASSERT_EQ(matr.size_padded(), 20);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size_padded(), [](const real_type val) { return val == real_type{ 0.0 }; }));
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorInvalidColumnsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with mismatching column sizes
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ { { real_type{ 0.0 }, real_type{ 0.0 } },
                                                            { real_type{ 0.0 } } },
                                                          plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorEmptyColumnsAndPadding) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with empty columns
    const std::vector<std::vector<real_type>> matr{ {}, {} };
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ matr, plssvm::shape{ 4, 5 } }),
                      plssvm::matrix_exception,
                      "The data to create the matrix must at least have one column!");
}
