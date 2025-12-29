/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Test the plssvm::matrix class member and free functions.
 */

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::matrix_exception
#include "plssvm/matrix.hpp"
#include "plssvm/shape.hpp"  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{real_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"             // util::redirect_output

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ASSERT_EQ, SCOPED_TRACE, FAIL,
                          // ::testing::{Test, StaticAssertTypeEq}
#include "fmt/format.h"   // fmt::format

#include <algorithm>  // std::swap
#include <cstddef>    // std::size_t
#include <iostream>   // std::clog
#include <string>     // std::string
#include <tuple>      // std::ignore
#include <vector>     // std::vector

template <typename T>
class MatrixFunctions : public ::testing::Test,
                        public util::redirect_output<&std::clog> {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};

TYPED_TEST_SUITE(MatrixFunctions, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixFunctions, Size) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.size(), 168);
}

TYPED_TEST(MatrixFunctions, Shape) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 42, 4 }));
}

TYPED_TEST(MatrixFunctions, NumRows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.num_rows(), 42);
}

TYPED_TEST(MatrixFunctions, NumCols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.num_cols(), 4);
}

TYPED_TEST(MatrixFunctions, Empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_FALSE(matr.empty());
    EXPECT_TRUE((plssvm::matrix<real_type, layout>{}).empty());
}

TYPED_TEST(MatrixFunctions, Layout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 4, 4 } };

    // check getter
    EXPECT_EQ(matr.layout(), layout);
}

TYPED_TEST(MatrixFunctions, FunctionCallOperator) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr(row, col), matr_2D[row][col]);
        }
    }
}

TYPED_TEST(MatrixFunctions, FunctionCallOperatorConst) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr(row, col), matr_2D[row][col]);
        }
    }
}

TYPED_TEST(MatrixFunctions, At) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr.at(row, col), matr_2D[row][col]);
        }
    }
}

TYPED_TEST(MatrixFunctions, AtOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}

TYPED_TEST(MatrixFunctions, AtConst) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            EXPECT_EQ(matr.at(row, col), matr_2D[row][col]);
        }
    }
}

TYPED_TEST(MatrixFunctions, AtConstOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}

TYPED_TEST(MatrixFunctions, SubscriptOperator) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if constexpr (layout == plssvm::layout_type::aos) {
                EXPECT_EQ(matr[row * 2 + col], matr_2D[row][col]);
            } else {
                EXPECT_EQ(matr[col * 2 + row], matr_2D[row][col]);
            }
        }
    }
}

TYPED_TEST(MatrixFunctions, SubscriptOperatorConst) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if constexpr (layout == plssvm::layout_type::aos) {
                EXPECT_EQ(matr[row * 2 + col], matr_2D[row][col]);
            } else {
                EXPECT_EQ(matr[col * 2 + row], matr_2D[row][col]);
            }
        }
    }
}

TYPED_TEST(MatrixFunctions, LinearAt) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if constexpr (layout == plssvm::layout_type::aos) {
                EXPECT_EQ(matr.at(row * 2 + col), matr_2D[row][col]);
            } else {
                EXPECT_EQ(matr.at(col * 2 + row), matr_2D[row][col]);
            }
        }
    }
}

TYPED_TEST(MatrixFunctions, LinearAtOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(4), plssvm::matrix_exception, "The current index (4) must be smaller than the total number of matrix entries (4)!");
}

TYPED_TEST(MatrixFunctions, LinearAtConst) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    ASSERT_EQ(matr.shape(), (plssvm::shape{ 2, 2 }));
    for (std::size_t row = 0; row < matr.num_rows(); ++row) {
        for (std::size_t col = 0; col < matr.num_cols(); ++col) {
            SCOPED_TRACE(fmt::format("row: {}; col: {}", row, col));
            if constexpr (layout == plssvm::layout_type::aos) {
                EXPECT_EQ(matr.at(row * 2 + col), matr_2D[row][col]);
            } else {
                EXPECT_EQ(matr.at(col * 2 + row), matr_2D[row][col]);
            }
        }
    }
}

TYPED_TEST(MatrixFunctions, LinearAtConstOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(4), plssvm::matrix_exception, "The current index (4) must be smaller than the total number of matrix entries (4)!");
}

TYPED_TEST(MatrixFunctions, To2DVector) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create the 2D vector
    const std::vector<std::vector<real_type>> matr_2D{ { real_type{ 0.1 }, real_type{ 0.2 } },
                                                       { real_type{ 1.1 }, real_type{ 1.2 } } };

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check content
    EXPECT_EQ(matr.to_2D_vector(), matr_2D);
}

TYPED_TEST(MatrixFunctions, SwapMemberFunction) {
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
    ASSERT_EQ(matr1.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr1.size(), 6);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr2.size(), 6);
    EXPECT_EQ(matr2.to_2D_vector(), matr1_2D);
}

TYPED_TEST(MatrixFunctions, SwapFreeFunction) {
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
    ASSERT_EQ(matr1.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr1.size(), 6);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr2.size(), 6);
    EXPECT_EQ(matr2.to_2D_vector(), matr1_2D);
}

TYPED_TEST(MatrixFunctions, OperatorEqual) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create matrices
    const plssvm::matrix<real_type, layout> matr1{ plssvm::shape{ 3, 2 } };
    const plssvm::matrix<real_type, layout> matr2{ plssvm::shape{ 2, 3 } };
    const plssvm::matrix<real_type, layout> matr3{ plssvm::shape{ 3, 3 } };
    const plssvm::matrix<real_type, layout> matr4{ plssvm::shape{ 3, 3 }, real_type{ 3.1415 } };
    const plssvm::matrix<real_type, layout> matr5{ plssvm::shape{ 3, 3 }, real_type{} };

    // check for equality
    EXPECT_FALSE(matr1 == matr2);
    EXPECT_FALSE(matr1 == matr3);
    EXPECT_FALSE(matr2 == matr3);
    EXPECT_FALSE(matr3 == matr4);
    EXPECT_TRUE(matr3 == matr5);
    EXPECT_TRUE(matr4 == matr4);
}

TYPED_TEST(MatrixFunctions, OperatorUnequal) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create matrices
    const plssvm::matrix<real_type, layout> matr1{ plssvm::shape{ 3, 2 } };
    const plssvm::matrix<real_type, layout> matr2{ plssvm::shape{ 2, 3 } };
    const plssvm::matrix<real_type, layout> matr3{ plssvm::shape{ 3, 3 } };
    const plssvm::matrix<real_type, layout> matr4{ plssvm::shape{ 3, 3 }, real_type{ 3.1415 } };
    const plssvm::matrix<real_type, layout> matr5{ plssvm::shape{ 3, 3 }, real_type{} };

    // check for equality
    EXPECT_TRUE(matr1 != matr2);
    EXPECT_TRUE(matr1 != matr3);
    EXPECT_TRUE(matr2 != matr3);
    EXPECT_TRUE(matr3 != matr4);
    EXPECT_FALSE(matr3 != matr5);
    EXPECT_FALSE(matr4 != matr4);
}

TYPED_TEST(MatrixFunctions, OutputOperator) {
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

TYPED_TEST(MatrixFunctions, Formatter) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create data
    const std::vector<std::vector<real_type>> matr_2D = { { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } },
                                                          { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } } };

    // create matrix
    const plssvm::matrix<real_type, layout> matr{ matr_2D };

    // check
    std::string correct_output{};
    for (const std::vector<real_type> &row : matr_2D) {
        correct_output += fmt::format("{:.10e} \n", fmt::join(row, " "));
    }
    correct_output.pop_back();  // remove last newline
    EXPECT_EQ(fmt::format("{}", matr), correct_output);
}

TYPED_TEST(MatrixFunctions, MatrixShorthands) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    if constexpr (layout == plssvm::layout_type::aos) {
        ::testing::StaticAssertTypeEq<plssvm::matrix<real_type, layout>, plssvm::aos_matrix<real_type>>();
    } else if constexpr (layout == plssvm::layout_type::soa) {
        ::testing::StaticAssertTypeEq<plssvm::matrix<real_type, layout>, plssvm::soa_matrix<real_type>>();
    } else {
        FAIL() << "Unrecognized layout type!";
    }
}

template <typename T>
class MatrixFunctionsDeathTest : public MatrixFunctions<T> { };

TYPED_TEST_SUITE(MatrixFunctionsDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixFunctionsDeathTest, FunctionCallOperatorOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}

TYPED_TEST(MatrixFunctionsDeathTest, FunctionCallOperatorConstOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}

TYPED_TEST(MatrixFunctionsDeathTest, SubscriptOperatorOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr[4], ::testing::HasSubstr("The current index (4) must be smaller than the total number of matrix entries (4)!"));
}

TYPED_TEST(MatrixFunctionsDeathTest, SubscriptOperatorConstOutOfBounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr[4], ::testing::HasSubstr("The current index (4) must be smaller than the total number of matrix entries (4)!"));
}