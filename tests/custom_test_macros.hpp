/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Header defining custom assertion macros.
 */

#ifndef PLSSVM_TESTS_CUSTOM_TEST_MACROS_HPP_
#define PLSSVM_TESTS_CUSTOM_TEST_MACROS_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"       // PLSSVM_ASSERT
#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::{always_false_v, remove_cvref_t, is_optional_v, is_reference_wrapper_v}, PLSSVM_REQUIRES
#include "plssvm/matrix.hpp"              // plssvm::matrix

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::{StrEq, Ge, Le, Gt, Lt}
#include "gtest/gtest.h"  // EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, ASSERT_FLOAT_EQ, ASSERT_DOUBLE_EQ, EXPECT_EQ, ASSERT_EQ, FAIL, EXPECT_LT, ASSERT_LT

#include <algorithm>    // std::max, std::min
#include <cmath>        // std::abs
#include <limits>       // std::numeric_limits::{epsilon, max, min}
#include <sstream>      // std::ostringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace detail {

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `EXPECT_FLOAT_EQ` and `ASSERT_DOUBLE_EQ` if @p expect is `true`, otherwise wraps `ASSERT_FLOAT_EQ` and `ASSERT_DOUBLE_EQ`.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 first value to compare (the actual value)
 * @param[in] val2 second value to compare (the expected value)
 * @param[in] msg an optional message
 */
template <bool expect, typename T>
inline void floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
    if constexpr (std::is_same_v<plssvm::detail::remove_cvref_t<T>, float>) {
        if constexpr (expect) {
            EXPECT_FLOAT_EQ(val1, val2) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
        } else {
            ASSERT_FLOAT_EQ(val1, val2) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
        }
    } else if constexpr (std::is_same_v<plssvm::detail::remove_cvref_t<T>, double>) {
        if constexpr (expect) {
            EXPECT_DOUBLE_EQ(val1, val2) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
        } else {
            ASSERT_DOUBLE_EQ(val1, val2) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
        }
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "T must be either float or double!");
    }
}

/**
 * @brief Compares the two vectors of floating point values @p val1 and @p val2.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 the first vector to compare (the actual value)
 * @param[in] val2 the second vector to compare (the expected value)
 */
template <bool expect, typename T>
inline void floating_point_vector_eq(const std::vector<T> &val1, const std::vector<T> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_eq<expect>(val1[col], val2[col], fmt::format("values at [{}] are not equal: ", col));
    }
}

/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 the first 2D vector to compare (the actual value)
 * @param[in] val2 the second 2D vector to compare (the expected value)
 */
template <bool expect, typename T>
inline void floating_point_2d_vector_eq(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_eq<expect>(val1[row][col], val2[row][col], fmt::format("values at [{}][{}] are not equal: ", row, col));
        }
    }
}

/**
 * @brief Compares the two matrices of floating point values @p matr1 and @p matr2.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam matrix_type the matrix type (AoS vs SoA)
 * @param[in] matr1 the first matrix to compare (the actual value)
 * @param[in] matr2 the second matrix to compare (the expected value)
 */
template <bool expect, typename matrix_type>
inline void floating_point_matrix_eq(const matrix_type &matr1, const matrix_type &matr2) {
    ASSERT_EQ(matr1.shape(), matr2.shape());
    ASSERT_EQ(matr1.padding(), matr2.padding());
    for (typename matrix_type::size_type row = 0; row < matr1.num_rows_padded(); ++row) {
        for (typename matrix_type::size_type col = 0; col < matr1.num_cols_padded(); ++col) {
            floating_point_eq<expect>(matr1(row, col), matr2(row, col), fmt::format("values at [{}][{}] are not equal: ", row, col));
        }
    }
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 first value to compare (the actual value)
 * @param[in] val2 second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 * @param[in] msg an optional message
 */
template <bool expect, typename T>
inline void floating_point_near(const T val1, const T val2, const T eps_factor = T{ 128.0 }, const std::string &msg = "") {
    // based on: https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison

    // set epsilon
    const T eps = eps_factor * std::numeric_limits<T>::epsilon();

    // sanity checks for picked epsilon value
    PLSSVM_ASSERT(std::numeric_limits<T>::epsilon() <= eps, "Chosen epsilon too small!: {} < {}", eps, std::numeric_limits<T>::epsilon());
    PLSSVM_ASSERT(eps < T{ 1.0 }, "Chosen epsilon too large!: {} >= 1.0", eps);

    if (val1 == val2) {
        SUCCEED();
    }

    const T diff = std::abs(val1 - val2);
    const T norm = std::min((std::abs(val1) + std::abs(val2)), std::numeric_limits<T>::max());

    if constexpr (expect) {
        EXPECT_LT(diff, std::max(std::numeric_limits<T>::min(), eps * norm)) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
    } else {
        ASSERT_LT(diff, std::max(std::numeric_limits<T>::min(), eps * norm)) << fmt::format("{}{} (actual) vs {} (expected)", msg, val1, val2);
    }
}

/**
 * @brief Compares the two vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 the first vector to compare (the actual value)
 * @param[in] val2 the second vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <bool expect, typename T>
inline void floating_point_vector_near(const std::vector<T> &val1, const std::vector<T> &val2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_near<expect>(val1[col], val2[col], eps_factor, fmt::format("values at [{}] are not equal enough: ", col));
    }
}

/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] val1 the first 2D vector to compare (the actual value)
 * @param[in] val2 the second 2D vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <bool expect, typename T>
inline void floating_point_2d_vector_near(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_near<expect>(val1[row][col], val2[row][col], eps_factor, fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
    }
}

/**
 * @brief Compares the two matrices @p matr1 and @p matr2 using a mixture of relative and absolute mode.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam matrix_type the matrix type (AoS vs SoA)
 * @tparam T the floating point type
 * @param[in] matr1 the first 2D vector to compare (the actual value)
 * @param[in] matr2 the second 2D vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <bool expect, typename matrix_type, typename T = typename matrix_type::value_type>
inline void floating_point_matrix_near(const matrix_type &matr1, const matrix_type &matr2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(matr1.shape(), matr2.shape());
    ASSERT_EQ(matr1.padding(), matr2.padding());
    for (typename matrix_type::size_type row = 0; row < matr1.num_rows_padded(); ++row) {
        for (typename matrix_type::size_type col = 0; col < matr2.num_cols_padded(); ++col) {
            floating_point_near<expect>(matr1(row, col), matr2(row, col), eps_factor, fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
    }
}

/**
 * @brief Tries to convert the @p value to a string using std::ostringstream. If it succeeds, compares the value to @p expected_str.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the type of the value to convert
 * @param[in] value the value to convert to a string
 * @param[in] expected_str the expected string representation of @p value
 */
template <bool expect, typename T>
inline void convert_to_string(const T &value, const std::string_view expected_str) {
    // convert value to a string
    std::ostringstream output;
    output << value;

    // test if the failbit has been set
    ASSERT_FALSE(output.fail());

    // check if the conversion was successful
    if constexpr (expect) {
        EXPECT_EQ(output.str(), expected_str);
    } else {
        ASSERT_EQ(output.str(), expected_str);
    }
}

/**
 * @brief Tries to convert the string @p str to a value of type T using std::istringstream. If it succeeds, compares the value to @p expected_value.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the type to which the string should be converted
 * @param[in] str the string to convert to a value of type T
 * @param[in] expected_value the expected value after conversion
 */
template <bool expect, typename T>
inline void convert_from_string(const std::string &str, const T &expected_value) {
    // convert a string to a value of type T
    std::istringstream input{ str };
    T value{};
    input >> value;

    // test if the failbit has been set
    ASSERT_FALSE(input.fail());

    // check if the conversion was successful
    if constexpr (expect) {
        EXPECT_EQ(value, expected_value);
    } else {
        ASSERT_EQ(value, expected_value);
    }
}

/**
 * @brief Checks whether the content of @p lhs, ignoring a potential std::reference_wrapper, is equal to @p rhs, also ignoring a potential std::reference_wrapper.
 * @details At least one of @p lhs or @p rhs must be an optional. If only one of them is an optional, which is empty, directly fails.
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the type contained in the first optional
 * @tparam U the type contained in the second optional
 * @param[in] lhs the first optional
 * @param[in] rhs the second optional
 */
template <bool expect, typename T, typename U, PLSSVM_REQUIRES(plssvm::detail::is_optional_v<T> || plssvm::detail::is_optional_v<U>)>
void expect_optional_equal(const T &lhs, const U &rhs) {
    // get the value of an optional, also supports std::optional<std::reference_wrapper<T>>
    const auto get_value = [](const auto &opt) {
        using value_type = typename plssvm::detail::remove_cvref_t<decltype(opt)>::value_type;
        if constexpr (plssvm::detail::is_reference_wrapper_v<value_type>) {
            return opt.value().get();
        } else {
            return opt.value();
        }
    };

    // case 1: lhs and rhs are both optionals
    // case 2: lhs is an optional, but rhs not
    // case 3: lhs is no optional, but rhs is
    // case 4: no optional -> can't happen due to template constraints
    if constexpr (plssvm::detail::is_optional_v<T> && plssvm::detail::is_optional_v<U>) {
        // if both optionals are empty, they are equal
        if (!lhs.has_value() && !rhs.has_value()) {
            SUCCEED();
        } else {
            // check the actual content
            if constexpr (expect) {
                EXPECT_EQ(get_value(lhs), get_value(rhs));
            } else {
                ASSERT_EQ(get_value(lhs), get_value(rhs));
            }
        }
    } else if constexpr (plssvm::detail::is_optional_v<T> && !plssvm::detail::is_optional_v<U>) {
        // the optional is empty, so there are unequal
        if (!lhs.has_value()) {
            FAIL() << "Provided optional does not contain a value!";
        }

        // check the actual content
        if constexpr (expect) {
            EXPECT_EQ(get_value(lhs), rhs);
        } else {
            ASSERT_EQ(get_value(lhs), rhs);
        }
    } else if constexpr (!plssvm::detail::is_optional_v<T> && plssvm::detail::is_optional_v<U>) {
        // call same function with reversed parameters
        expect_optional_equal<expect>(rhs, lhs);
    } else {
        // unreachable
        FAIL();
    }
}

}  // namespace detail

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_EQ(val1, val2) \
    ::detail::floating_point_eq<true>(val1, val2)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_EQ(val1, val2) \
    ::detail::floating_point_eq<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_VECTOR_EQ(val1, val2) \
    ::detail::floating_point_vector_eq<true>(val1, val2)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_EQ(val1, val2) \
    ::detail::floating_point_vector_eq<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_EQ(val1, val2) \
    ::detail::floating_point_2d_vector_eq<true>(val1, val2)
/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_EQ(val1, val2) \
    ::detail::floating_point_2d_vector_eq<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the matrix @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_MATRIX_EQ(val1, val2) \
    ::detail::floating_point_matrix_eq<true>(val1, val2)
/**
 * @brief Check whether the floating point values in the matrix @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_MATRIX_EQ(val1, val2) \
    ::detail::floating_point_matrix_eq<false>(val1, val2)

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_NEAR(val1, val2) \
    ::detail::floating_point_near<true>(val1, val2)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_NEAR(val1, val2) \
    ::detail::floating_point_near<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_VECTOR_NEAR(val1, val2) \
    ::detail::floating_point_vector_near<true>(val1, val2)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_NEAR(val1, val2) \
    ::detail::floating_point_vector_near<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2) \
    ::detail::floating_point_2d_vector_near<true>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2) \
    ::detail::floating_point_2d_vector_near<false>(val1, val2)

/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_MATRIX_NEAR(val1, val2) \
    ::detail::floating_point_matrix_near<true>(val1, val2)
/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_MATRIX_NEAR(val1, val2) \
    ::detail::floating_point_matrix_near<false>(val1, val2)

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_near<true>(val1, val2, eps_factor)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_near<false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_vector_near<true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_vector_near<false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_2d_vector_near<true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_2d_vector_near<false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_matrix_near<true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_MATRIX_NEAR_EPS(val1, val2, eps_factor) \
    ::detail::floating_point_matrix_near<false>(val1, val2, eps_factor)

/**
 * @brief Tries to convert the @p val to a string. If it succeeds, compares the value to @p str.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val the value to convert to a string
 * @param[in] str the expected string representation of @p value
 */
#define EXPECT_CONVERSION_TO_STRING(val, str) \
    ::detail::convert_to_string<true>(val, str)
/**
 * @brief Tries to convert the @p val to a string. If it succeeds, compares the value to @p str.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val the value to convert to a string
 * @param[in] str the expected string representation of @p value
 */
#define ASSERT_CONVERSION_TO_STRING(val, str) \
    ::detail::convert_to_string<false>(val, str)

/**
 * @brief Tries to convert the string @p str to a value of type `decltype(T)`. If it succeeds, compares the value to @p val.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] str the string to convert to a value of type T
 * @param[in] val the expected value after conversion
 */
#define EXPECT_CONVERSION_FROM_STRING(str, val) \
    ::detail::convert_from_string<true>(str, val)
/**
 * @brief Tries to convert the string @p str to a value of type `decltype(T)`. If it succeeds, compares the value to @p val.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] str the string to convert to a value of type T
 * @param[in] val the expected value after conversion
 */
#define ASSERT_CONVERSION_FROM_STRING(str, val) \
    ::detail::convert_from_string<false>(str, val)

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while): the idiomatic C++ way to ensure that the macro behaves properly with semicolons
/**
 * @brief Check whether @p statement throws an exception of type @p expected_exception and the exception's `what()` message matches the GTest @p matcher.
 * @details Succeeds only if the exception type **and** message match.
 * @param[in] statement the statement that should throw an exception
 * @param[in] expected_exception the type of the exception that should be thrown
 * @param[in] matcher the GtTest matcher used to test the exception's `what()` message
 */
#define EXPECT_THROW_WHAT_MATCHER(statement, expected_exception, matcher)                                    \
    do {                                                                                                     \
        try {                                                                                                \
            statement;                                                                                       \
            FAIL() << "Expected " #expected_exception;                                                       \
        } catch (const expected_exception &e) {                                                              \
            EXPECT_THAT(std::string_view(e.what()), matcher);                                                \
        } catch (...) {                                                                                      \
            FAIL() << "The expected exception type (" #expected_exception ") doesn't match the caught one!"; \
        }                                                                                                    \
    } while (false)
// NOLINTEND(cppcoreguidelines-avoid-do-while)

/**
 * @brief Check whether @p statement throws an exception of type @p expected_exception with the exception's `what()` message @p msg.
 * @details Succeeds only if the exception type **and** message match.
 * @param[in] statement the statement that should throw an exception
 * @param[in] expected_exception the type of the exception that should be thrown
 * @param[in] msg the expected exception's `what()` message
 */
#define EXPECT_THROW_WHAT(statement, expected_exception, msg) EXPECT_THROW_WHAT_MATCHER(statement, expected_exception, ::testing::StrEq(msg))

// NOLINTBEGIN(cppcoreguidelines-avoid-do-while): the idiomatic C++ way to ensure that the macro behaves properly with semicolons
/**
 * @brief Check whether the value of @p instance is an instance of the @p type.
 * @param[in] type the type the @p instance should have, assumed to not be a pointer type
 * @param[in] instance the instance to check, assumed to be a pointer type
 */
#define EXPECT_INSTANCE_OF(type, instance)           \
    do {                                             \
        auto ptr = dynamic_cast<type *>(&*instance); \
        EXPECT_NE(ptr, nullptr);                     \
    } while (false)
// NOLINTEND(cppcoreguidelines-avoid-do-while)

/**
 * @brief Check whether @p val is in the **inclusive** range [@p min, @p max].
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val the value to check
 * @param[in] min the lower bound value for @p val (**inclusive**)
 * @param[in] max the upper bound value for @p val (**inclusive**)
 */
#define EXPECT_INCLUSIVE_RANGE(val, min, max) EXPECT_THAT((val), ::testing::AllOf(::testing::Ge((min)), ::testing::Le((max))))
/**
 * @brief Check whether @p val is in the **inclusive** range [@p min, @p max].
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val the value to check
 * @param[in] min the lower bound value for @p val (**inclusive**)
 * @param[in] max the upper bound value for @p val (**inclusive**)
 */
#define ASSERT_INCLUSIVE_RANGE(val, min, max) ASSERT_THAT((val), ::testing::AllOf(::testing::Ge((min)), ::testing::Le((max))))
/**
 * @brief Check whether @p val is in the **exclusive** range (@p min, @p max).
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val the value to check
 * @param[in] min the lower bound value for @p val (**exclusive**)
 * @param[in] max the upper bound value for @p val (**exclusive**)
 */
#define EXPECT_EXCLUSIVE_RANGE(val, min, max) EXPECT_THAT((val), ::testing::AllOf(::testing::Gt((min)), ::testing::Lt((max))))
/**
 * @brief Check whether @p val is in the **exclusive** range (@p min, @p max).
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val the value to check
 * @param[in] min the lower bound value for @p val (**exclusive**)
 * @param[in] max the upper bound value for @p val (**exclusive**)
 */
#define ASSERT_EXCLUSIVE_RANGE(val, min, max) ASSERT_THAT((val), ::testing::AllOf(::testing::Gt((min)), ::testing::Lt((max))))

/**
 * @brief Checks whether the content of @p val1 is equal to @p val2. Both or one of both can be a std::optional (potentially containing a std::reference_wrapper).
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the optional to check (must be an optional)
 * @param[in] val2 the value to check the optional against (may also be an optional, but not necessarily)
 */
#define EXPECT_OPTIONAL_EQ(val1, val2) ::detail::expect_optional_equal<false>(val1, val2)
/**
 * @brief Checks whether the content of @p val1 is equal to @p val2. Both or one of both can be a std::optional (potentially containing a std::reference_wrapper).
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the optional to check (must be an optional)
 * @param[in] val2 the value to check the optional against (may also be an optional, but not necessarily)
 */
#define ASSERT_OPTIONAL_EQ(val1, val2) ::detail::expect_optional_equal<true>(val1, val2)

#endif  // PLSSVM_TESTS_CUSTOM_TEST_MACROS_HPP_
