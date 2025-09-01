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
#include "plssvm/shape.hpp"                  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{real_type_layout_type_gtest, test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"             // util::{generate_random_matrix, redirect_output}

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ASSERT_EQ, SCOPED_TRACE, FAIL,
                          // ::testing::{Test, StaticAssertTypeEq}
#include "fmt/format.h"   // fmt::format

#include <algorithm>  // std::swap
#include <cstddef>    // std::size_t
#include <iostream>   // std::clog
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
class Matrix : public ::testing::Test,
               public util::redirect_output<&std::clog> {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::layout_type fixture_layout = util::test_parameter_value_at_v<0, T>;
};

TYPED_TEST_SUITE(Matrix, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(Matrix, construct_default) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // default construct matrix
    const plssvm::matrix<real_type, layout> matr{};

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_with_size) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 3, 2 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr.size(), 6);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size(), [](const real_type val) { return val == real_type{}; }));
}

TYPED_TEST(Matrix, construct_with_size_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_with_size_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(Matrix, construct_with_size_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_default_value) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 3 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr.size(), 6);
    EXPECT_TRUE(std::all_of(matr.data(), matr.data() + matr.size(), [](const real_type val) { return val == real_type{ 3.1415 }; }));
}

TYPED_TEST(Matrix, construct_with_size_and_default_value_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, real_type{ 3.1415 } };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_with_size_and_default_value_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_default_value_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, real_type{ 3.1415 } }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_vector) {
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

TYPED_TEST(Matrix, construct_with_size_and_vector_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, std::vector<real_type>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_with_size_and_vector_value_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_vector_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_vector_size_mismatch) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 2 }, std::vector<real_type>(2) }),
                      plssvm::matrix_exception,
                      "The number of entries in the matrix (4) must be equal to the size of the data (2)!");
}

TYPED_TEST(Matrix, construct_with_size_and_pointer) {
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

TYPED_TEST(Matrix, construct_with_size_and_ptr_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size
    const std::vector<real_type> data{};
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 0, 0 }, data.data() };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_with_size_and_nullptr_ptr) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix with a specific size and a nullptr
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 3 }, nullptr }),
                      plssvm::matrix_exception,
                      "The provided data pointer may not be a nullptr if the matrix size is greater than 0!");
}

TYPED_TEST(Matrix, construct_with_size_and_ptr_value_zero_num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero rows
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 0, 2 }, data.data() }),
                      plssvm::matrix_exception,
                      "The number of rows is zero but the number of columns is not!");
}

TYPED_TEST(Matrix, construct_with_size_and_ptr_zero_num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix with zero columns
    const std::vector<real_type> data(2);
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ plssvm::shape{ 2, 0 }, data.data() }),
                      plssvm::matrix_exception,
                      "The number of columns is zero but the number of rows is not!");
}

TYPED_TEST(Matrix, construct_from_same_matrix_layout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct matrix
    const auto matr = util::generate_random_matrix<plssvm::matrix<real_type, layout>>(plssvm::shape{ 3, 2 });

    // create a new matrix from this matrix
    const plssvm::matrix<real_type, layout> new_matr{ matr };

    // both matrices should be identical
    EXPECT_EQ(new_matr, matr);
}

TYPED_TEST(Matrix, construct_from_other_matrix_layout) {
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

TYPED_TEST(Matrix, construct_from_2D_vector) {
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

TYPED_TEST(Matrix, construct_from_2D_vector_empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // construct a matrix from a std::vector<std::vector<>>
    const plssvm::matrix<real_type, layout> matr{ std::vector<std::vector<real_type>>{} };

    // check content
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 0, 0 }));
}

TYPED_TEST(Matrix, construct_from_2D_vector_invalid_columns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with mismatching column sizes
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ { { real_type{ 0.0 }, real_type{ 0.0 } },
                                                            { real_type{ 0.0 } } } }),
                      plssvm::matrix_exception,
                      "Each row in the matrix must contain the same amount of columns!");
}

TYPED_TEST(Matrix, construct_from_2D_vector_empty_columns) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // try constructing a matrix from an empty 2D vector with empty columns
    const std::vector<std::vector<real_type>> matr{ {}, {} };
    EXPECT_THROW_WHAT((plssvm::matrix<real_type, layout>{ matr }),
                      plssvm::matrix_exception,
                      "The data to create the matrix must at least have one column!");
}

TYPED_TEST(Matrix, size) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.size(), 168);
}

TYPED_TEST(Matrix, shape) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.shape(), (plssvm::shape{ 42, 4 }));
}

TYPED_TEST(Matrix, num_rows) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.num_rows(), 42);
}

TYPED_TEST(Matrix, num_cols) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_EQ(matr.num_cols(), 4);
}

TYPED_TEST(Matrix, empty) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 42, 4 } };
    // check getter
    EXPECT_FALSE(matr.empty());
    EXPECT_TRUE((plssvm::matrix<real_type, layout>{}).empty());
}


TYPED_TEST(Matrix, layout) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 4, 4 } };

    // check getter
    EXPECT_EQ(matr.layout(), layout);
}

TYPED_TEST(Matrix, function_call_operator) {
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

TYPED_TEST(Matrix, function_call_operator_const) {
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

TYPED_TEST(Matrix, at) {
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

TYPED_TEST(Matrix, at_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}

TYPED_TEST(Matrix, at_const) {
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

TYPED_TEST(Matrix, at_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(3, 0), plssvm::matrix_exception, "The current row (3) must be smaller than the number of rows (2)!");
    EXPECT_THROW_WHAT(std::ignore = matr.at(0, 2), plssvm::matrix_exception, "The current column (2) must be smaller than the number of columns (2)!");
}

TYPED_TEST(Matrix, subscript_operator) {
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

TYPED_TEST(Matrix, subscript_operator_const) {
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

TYPED_TEST(Matrix, linear_at) {
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

TYPED_TEST(Matrix, linear_at_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(4), plssvm::matrix_exception, "The current index (4) must be smaller than the total number of matrix entries (4)!");
}

TYPED_TEST(Matrix, linear_at_const) {
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

TYPED_TEST(Matrix, linear_at_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_THROW_WHAT(std::ignore = matr.at(4), plssvm::matrix_exception, "The current index (4) must be smaller than the total number of matrix entries (4)!");
}

TYPED_TEST(Matrix, to_2D_vector) {
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
    ASSERT_EQ(matr1.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr1.size(), 6);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr2.size(), 6);
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
    ASSERT_EQ(matr1.shape(), (plssvm::shape{ 3, 2 }));
    ASSERT_EQ(matr1.size(), 6);
    EXPECT_EQ(matr1.to_2D_vector(), matr2_2D);

    // check the content of matr2
    ASSERT_EQ(matr2.shape(), (plssvm::shape{ 2, 3 }));
    ASSERT_EQ(matr2.size(), 6);
    EXPECT_EQ(matr2.to_2D_vector(), matr1_2D);
}

TYPED_TEST(Matrix, operator_equal) {
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

TYPED_TEST(Matrix, operator_unequal) {
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

TYPED_TEST(Matrix, formatter) {
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

//*************************************************************************************************************************************//
//                                                      plssvm::matrix operations                                                      //
//*************************************************************************************************************************************//
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

TYPED_TEST(MatrixOperations, operator_scale_binary) {
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

TYPED_TEST(MatrixOperations, operator_scale_binary_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() * this->get_scalar(), this->get_empty());
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_scalar() * this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, operator_scale_compound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // scale a matrix using a scalar
    const plssvm::matrix<real_type, layout> C{ { { 1.5, 3.0, 4.5 }, { 6.0, 7.5, 9.0 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() *= this->get_scalar(), C);
}

TYPED_TEST(MatrixOperations, operator_scale_compound_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() *= this->get_scalar(), this->get_empty());
}

TYPED_TEST(MatrixOperations, operator_add_binary) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { 2.5, 4.5, 6.5 }, { 8.5, 10.5, 12.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() + this->get_B(), C);
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_B() + this->get_A(), C);
}

TYPED_TEST(MatrixOperations, operator_add_binary_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() + this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, operator_add_compound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { 2.5, 4.5, 6.5 }, { 8.5, 10.5, 12.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() += this->get_B(), C);
}

TYPED_TEST(MatrixOperations, operator_add_compound_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() += this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, operator_add_binary) {
    // try adding two matrices with mismatching sizes
    EXPECT_DEATH(std::ignore = this->get_A() + this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(std::ignore = this->get_B() + this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperationsDeathTest, operator_add_compound) {
    // try adding two matrices with mismatching sizes
    EXPECT_DEATH(this->get_A() += this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(this->get_B() += this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperations, operator_subtract_binary) {
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

TYPED_TEST(MatrixOperations, operator_subtract_binary_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() - this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperations, operator_subtract_compound) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // add two matrices to each other
    const plssvm::matrix<real_type, layout> C{ { { -0.5, -0.5, -0.5 }, { -0.5, -0.5, -0.5 } } };
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_A() -= this->get_B(), C);
}

TYPED_TEST(MatrixOperations, operator_subtract_compound_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() -= this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, operator_subtract_binary) {
    // try subtracting two matrices with mismatching sizes
    EXPECT_DEATH(std::ignore = this->get_A() - this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(std::ignore = this->get_B() - this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperationsDeathTest, operator_subtract_compound) {
    // try subtracting two matrices with mismatching sizes
    EXPECT_DEATH(this->get_A() -= this->get_B(), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
    EXPECT_DEATH(this->get_B() -= this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! ([3, 2] != [2, 3])"));
}

TYPED_TEST(MatrixOperations, operator_matrix_multiplication_square) {
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

TYPED_TEST(MatrixOperations, operator_matrix_multiplication) {
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

TYPED_TEST(MatrixOperations, operator_matrix_multiplication_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(this->get_empty() * this->get_empty(), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, operator_matrix_multiplication) {
    EXPECT_DEATH(std::ignore = this->get_A() * this->get_empty(), ::testing::HasSubstr("Error: shapes missmatch! (3 (num_cols) != 0 (num_rows))"));
    EXPECT_DEATH(std::ignore = this->get_empty() * this->get_A(), ::testing::HasSubstr("Error: shapes missmatch! (0 (num_cols) != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, operator_rowwise_dot) {
    using real_type = typename TestFixture::fixture_real_type;

    // rowwise dot product
    const std::vector<real_type> ret{ 17.0, 84.5 };
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_A(), this->get_B()), ret);
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_B(), this->get_A()), ret);
}

TYPED_TEST(MatrixOperations, operator_rowwise_dot_empty) {
    EXPECT_FLOATING_POINT_VECTOR_EQ(rowwise_dot(this->get_empty(), this->get_empty()), {});
}

TYPED_TEST(MatrixOperationsDeathTest, operator_rowwise_dot) {
    // sizes missmatch
    EXPECT_DEATH(std::ignore = rowwise_dot({}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! ([0, 0] != [2, 3])"));
    EXPECT_DEATH(std::ignore = rowwise_dot(this->get_A(), this->get_B()), ::testing::HasSubstr("Error: shapes missmatch! ([2, 3] != [3, 2])"));
}

TYPED_TEST(MatrixOperations, operator_rowwise_scale) {
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

TYPED_TEST(MatrixOperations, operator_rowwise_scale_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(rowwise_scale({}, this->get_empty()), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, operator_rowwise_scale) {
    using real_type = typename TestFixture::fixture_real_type;

    // sizes missmatch
    EXPECT_DEATH(std::ignore = rowwise_scale({}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = rowwise_scale(std::vector<real_type>{ 1.0, 2.0, 3.0 }, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, operator_masked_rowwise_scale) {
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

TYPED_TEST(MatrixOperations, operator_masked_rowwise_scale_empty) {
    EXPECT_FLOATING_POINT_MATRIX_EQ(masked_rowwise_scale({}, {}, this->get_empty()), this->get_empty());
}

TYPED_TEST(MatrixOperationsDeathTest, operator_masked_rowwise_scale) {
    using real_type = typename TestFixture::fixture_real_type;

    const std::vector<real_type> scale{ 1.0, 2.0 };
    const std::vector<unsigned long long> mask{ 1, 0 };

    // sizes missmatch
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(mask, {}, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(mask, std::vector<real_type>{ 1.0, 2.0, 3.0 }, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(std::vector<unsigned long long>{}, scale, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (0 != 2 (num_rows))"));
    EXPECT_DEATH(std::ignore = masked_rowwise_scale(std::vector<unsigned long long>{ 1, 0, 1 }, scale, this->get_A()), ::testing::HasSubstr("Error: shapes missmatch! (3 != 2 (num_rows))"));
}

TYPED_TEST(MatrixOperations, operator_variance) {
    using real_type = typename TestFixture::fixture_real_type;

    EXPECT_FLOATING_POINT_NEAR(variance(this->get_A()), (real_type{ 17.5 } / static_cast<real_type>(this->get_A().size())));
    EXPECT_FLOATING_POINT_NEAR(variance(this->get_B()), (real_type{ 17.5 } / static_cast<real_type>(this->get_A().size())));
}

TYPED_TEST(Matrix, matrix_shorthands) {
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
class MatrixDeathTest : public Matrix<T> { };

TYPED_TEST_SUITE(MatrixDeathTest, util::real_type_layout_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(MatrixDeathTest, function_call_operator_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}

TYPED_TEST(MatrixDeathTest, function_call_operator_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr(3, 0), ::testing::HasSubstr("The current row (3) must be smaller than the number of rows (2)!"));
    EXPECT_DEATH(std::ignore = matr(0, 2), ::testing::HasSubstr("The current column (2) must be smaller than the number of columns (2)!"));
}

TYPED_TEST(MatrixDeathTest, subscript_operator_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr[4], ::testing::HasSubstr("The current index (4) must be smaller than the total number of matrix entries (4)!"));
}

TYPED_TEST(MatrixDeathTest, subscript_operator_const_out_of_bounce) {
    using real_type = typename TestFixture::fixture_real_type;
    constexpr plssvm::layout_type layout = TestFixture::fixture_layout;

    // create random matrix
    const plssvm::matrix<real_type, layout> matr{ plssvm::shape{ 2, 2 } };

    // try out-of-bounce access
    EXPECT_DEATH(std::ignore = matr[4], ::testing::HasSubstr("The current index (4) must be smaller than the total number of matrix entries (4)!"));
}
