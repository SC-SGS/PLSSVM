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
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
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

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValue) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr.size(), 6);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size(), [](const real_type val) { return val == real_type{ 3.1415 }; }));
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndDefaultValueEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
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
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndVectorEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, std::vector<real_type>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
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
}

TYPED_TEST(MatrixConstructors, ConstructWithSizeAndPointerEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data{};
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, data.data() };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
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

    // check content
    ASSERT_EQ(new_matr.size(), matr.size());
    for (std::size_t row = 0; row < new_matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < new_matr.num_cols(); ++col) {
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
    ASSERT_EQ(matr.size(), 4);
    auto val = static_cast<real_type>(0.1);
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_FLOATING_POINT_EQ(matr(row, col), val);
            val += real_type{ 0.1 };
        }
    }
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);
}

TYPED_TEST(MatrixConstructors, ConstructFrom2DVectorEmpty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ std::vector<std::vector<real_type>>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
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
