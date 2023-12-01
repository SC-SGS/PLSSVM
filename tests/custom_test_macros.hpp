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

#include "plssvm/detail/assert.hpp"       // PLSSVM_ASSERT
#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::{always_false_v, remove_cvref_t}
#include "plssvm/matrix.hpp"              // plssvm::matrix

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{StrEq, Ge, Le, Gt, Lt}
#include "gtest/gtest.h"           // EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, ASSERT_FLOAT_EQ, ASSERT_DOUBLE_EQ, EXPECT_EQ, ASSERT_EQ, FAIL, EXPECT_LT, ASSERT_LT

#include <algorithm>    // std::max, std::min
#include <cmath>        // std::abs
#include <limits>       // std::numeric_limits::{epsilon, max, min}
#include <sstream>      // std::ostringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace detail {

// type_trait to get the `value_type` of a `std::vector<T>`, `std::vector<std::vector<T>>`, etc.
template <typename T>
struct get_value_type {
    using type = T;
};
template <typename T>
struct get_value_type<std::vector<T>> {
    using type = typename get_value_type<T>::type;
};
template <typename T>
using get_value_type_t = typename get_value_type<T>::type;

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `EXPECT_FLOAT_EQ` and `ASSERT_DOUBLE_EQ` if @p expect is `true`, otherwise wraps `ASSERT_FLOAT_EQ` and `ASSERT_DOUBLE_EQ`.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 first value to compare (the actual value)
 * @param[in] val2 second value to compare (the expected value)
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
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
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first vector to compare (the actual value)
 * @param[in] val2 the second vector to compare (the expected value)
 */
template <typename T, bool expect>
inline void floating_point_vector_eq(const std::vector<T> &val1, const std::vector<T> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_eq<T, expect>(val1[col], val2[col], fmt::format("values at [{}] are not equal: ", col));
    }
}
/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first 2D vector to compare (the actual value)
 * @param[in] val2 the second 2D vector to compare (the expected value)
 */
template <typename T, bool expect>
inline void floating_point_2d_vector_eq(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_eq<T, expect>(val1[row][col], val2[row][col], fmt::format("values at [{}][{}] are not equal: ", row, col));
        }
    }
}
/**
 * @brief Compares the two matrices of floating point values @p matr1 and @p matr2.
 * @tparam matrix_type the matrix type (AoS vs SoA)
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] matr1 the first matrix to compare (the actual value)
 * @param[in] matr2 the second matrix to compare (the expected value)
 */
template <typename matrix_type, bool expect>
inline void floating_point_matrix_eq(const matrix_type &matr1, const matrix_type &matr2) {
    ASSERT_EQ(matr1.shape(), matr2.shape());
    ASSERT_EQ(matr1.padding(), matr2.padding());
    for (typename matrix_type::size_type row = 0; row < matr1.num_rows_padded(); ++row) {
        for (typename matrix_type::size_type col = 0; col < matr1.num_cols_padded(); ++col) {
            floating_point_eq<typename matrix_type::value_type, expect>(matr1(row, col), matr2(row, col), fmt::format("values at [{}][{}] are not equal: ", row, col));
        }
    }
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 first value to compare (the actual value)
 * @param[in] val2 second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
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
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first vector to compare (the actual value)
 * @param[in] val2 the second vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <typename T, bool expect>
inline void floating_point_vector_near(const std::vector<T> &val1, const std::vector<T> &val2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_near<T, expect>(val1[col], val2[col], eps_factor, fmt::format("values at [{}] are not equal enough: ", col));
    }
}
/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first 2D vector to compare (the actual value)
 * @param[in] val2 the second 2D vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <typename T, bool expect>
inline void floating_point_2d_vector_near(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_near<T, expect>(val1[row][col], val2[row][col], eps_factor, fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
    }
}
/**
 * @brief Compares the two matrices @p matr1 and @p matr2 using a mixture of relative and absolute mode.
 * @tparam matrix_type the matrix type (AoS vs SoA)
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @tparam T the floating point type
 * @param[in] matr1 the first 2D vector to compare (the actual value)
 * @param[in] matr2 the second 2D vector to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
template <typename matrix_type, bool expect, typename T = typename matrix_type::value_type>
inline void floating_point_matrix_near(const matrix_type &matr1, const matrix_type &matr2, const T eps_factor = T{ 128.0 }) {
    ASSERT_EQ(matr1.shape(), matr2.shape());
    ASSERT_EQ(matr1.padding(), matr2.padding());
    for (typename matrix_type::size_type row = 0; row < matr1.num_rows_padded(); ++row) {
        for (typename matrix_type::size_type col = 0; col < matr2.num_cols_padded(); ++col) {
            floating_point_near<T, expect>(matr1(row, col), matr2(row, col), eps_factor, fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
    }
}

/**
 * @brief Tries to convert the @p value to a string using std::ostringstream. If it succeeds, compares the value to @p expected_str.
 * @tparam T the type of the value to convert
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] value the value to convert to a string
 * @param[in] expected_str the expected string representation of @p value
 */
template <typename T, bool expect>
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
 * @tparam T the type to which the string should be converted
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] str the string to convert to a value of type T
 * @param[in] expected_value the expected value after conversion
 */
template <typename T, bool expect>
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

}  // namespace detail

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_EQ(val1, val2) \
    detail::floating_point_eq<decltype(val1), true>(val1, val2)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_EQ(val1, val2) \
    detail::floating_point_eq<decltype(val1), false>(val1, val2)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_VECTOR_EQ(val1, val2) \
    detail::floating_point_vector_eq<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_EQ(val1, val2) \
    detail::floating_point_vector_eq<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_EQ(val1, val2) \
    detail::floating_point_2d_vector_eq<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2)
/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_EQ(val1, val2) \
    detail::floating_point_2d_vector_eq<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

/**
 * @brief Check whether the floating point values in the matrix @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_MATRIX_EQ(val1, val2) \
    detail::floating_point_matrix_eq<plssvm::detail::remove_cvref_t<decltype(val1)>, true>(val1, val2)
/**
 * @brief Check whether the floating point values in the matrix @p val1 and @p val2 are "equal".
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_MATRIX_EQ(val1, val2) \
    detail::floating_point_matrix_eq<plssvm::detail::remove_cvref_t<decltype(val1)>, false>(val1, val2)

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_NEAR(val1, val2) \
    detail::floating_point_near<decltype(val1), true>(val1, val2)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_NEAR(val1, val2) \
    detail::floating_point_near<decltype(val1), false>(val1, val2)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_VECTOR_NEAR(val1, val2) \
    detail::floating_point_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_NEAR(val1, val2) \
    detail::floating_point_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2) \
    detail::floating_point_2d_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2) \
    detail::floating_point_2d_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define EXPECT_FLOATING_POINT_MATRIX_NEAR(val1, val2) \
    detail::floating_point_matrix_near<plssvm::detail::remove_cvref_t<decltype(val1)>, true>(val1, val2)
/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_MATRIX_NEAR(val1, val2) \
    detail::floating_point_matrix_near<plssvm::detail::remove_cvref_t<decltype(val1)>, false>(val1, val2)

/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_near<decltype(val1), true>(val1, val2, eps_factor)
/**
 * @brief Check whether the two floating point values @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_near<decltype(val1), false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_2D_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_2d_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the 2D vectors @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_2d_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2, eps_factor)

/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_matrix_near<plssvm::detail::remove_cvref_t<decltype(val1)>, true>(val1, val2, eps_factor)
/**
 * @brief Check whether the floating point values in the plssvm::matrix @p val1 and @p val2 are "equal enough" with respect to a mixture of a relative and absolute mode.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 * @param[in] eps_factor a scaling factor in the floating point near calculation
 */
#define ASSERT_FLOATING_POINT_MATRIX_NEAR_EPS(val1, val2, eps_factor) \
    detail::floating_point_matrix_near<plssvm::detail::remove_cvref_t<decltype(val1)>, false>(val1, val2, eps_factor)

/**
 * @brief Tries to convert the @p val to a string. If it succeeds, compares the value to @p str.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] val the value to convert to a string
 * @param[in] str the expected string representation of @p value
 */
#define EXPECT_CONVERSION_TO_STRING(val, str) \
    detail::convert_to_string<decltype(val), true>(val, str)
/**
 * @brief Tries to convert the @p val to a string. If it succeeds, compares the value to @p str.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] val the value to convert to a string
 * @param[in] str the expected string representation of @p value
 */
#define ASSERT_CONVERSION_TO_STRING(val, str) \
    detail::convert_to_string<decltype(val), false>(val, str)

/**
 * @brief Tries to convert the string @p str to a value of type `decltype(T)`. If it succeeds, compares the value to @p val.
 * @details Other tests in the test case are executed even if this test fails.
 * @param[in] str the string to convert to a value of type T
 * @param[in] val the expected value after conversion
 */
#define EXPECT_CONVERSION_FROM_STRING(str, val) \
    detail::convert_from_string<decltype(val), true>(str, val)
/**
 * @brief Tries to convert the string @p str to a value of type `decltype(T)`. If it succeeds, compares the value to @p val.
 * @details Other tests in the test case are aborted if this test fails.
 * @param[in] str the string to convert to a value of type T
 * @param[in] val the expected value after conversion
 */
#define ASSERT_CONVERSION_FROM_STRING(str, val) \
    detail::convert_from_string<decltype(val), false>(str, val)

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

/**
 * @brief Check whether @p statement throws an exception of type @p expected_exception with the exception's `what()` message @p msg.
 * @details Succeeds only if the exception type **and** message match.
 * @param[in] statement the statement that should throw an exception
 * @param[in] expected_exception the type of the exception that should be thrown
 * @param[in] msg the expected exception's `what()` message
 */
#define EXPECT_THROW_WHAT(statement, expected_exception, msg) EXPECT_THROW_WHAT_MATCHER(statement, expected_exception, ::testing::StrEq(msg))

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

#endif  // PLSSVM_TESTS_CUSTOM_TEST_MACROS_HPP_