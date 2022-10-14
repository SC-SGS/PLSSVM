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

#include "plssvm/detail/assert.hpp"   // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"  // plssvm::detail::always_false_v

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::StrEq
#include "gtest/gtest.h"           // EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, ASSERT_FLOAT_EQ, ASSERT_DOUBLE_EQ, EXPECT_EQ, ASSERT_EQ, SUCCESS, FAIL, EXPECT_LT, ASSERT_LT

#include <algorithm>    // std::max
#include <cmath>        // std::abs
#include <limits>       // std::numeric_limits::{epsilon, max, min}
#include <type_traits>  // std::is_same_v

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
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
    if constexpr (std::is_same_v<plssvm::detail::remove_cvref_t<T>, float>) {
        if constexpr (expect) {
            EXPECT_FLOAT_EQ(val1, val2) << fmt::format("{}{} (expected) vs {} (actual)", msg, val2, val1);
        } else {
            ASSERT_FLOAT_EQ(val1, val2) << fmt::format("{}{} (expected) vs {} (actual)", msg, val2, val1);
        }
    } else if constexpr (std::is_same_v<plssvm::detail::remove_cvref_t<T>, double>) {
        if constexpr (expect) {
            EXPECT_DOUBLE_EQ(val1, val2) << fmt::format("{}{} (expected) vs {} (actual)", msg, val2, val1);
        } else {
            ASSERT_DOUBLE_EQ(val1, val2) << fmt::format("{}{} (expected) vs {} (actual)", msg, val2, val1);
        }
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "T must be either float or double!");
    }
}

/**
 * @brief Compares the two vectors of floating point values @p val1 and @p val2.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first vector to compare
 * @param[in] val2 the second vector to compare
 * @param[in] msg an optional message
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
 * @param[in] val1 the first 2D vector to compare
 * @param[in] val2 the second 2D vector to compare
 * @param[in] msg an optional message
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
 * @brief Compares the two floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] scale scale the epsilon value by the provided value
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_near(const T val1, const T val2, const std::string &msg = "") {
    // based on: https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison

    // set epsilon
    const T eps = 128 * std::numeric_limits<T>::epsilon();

    // sanity checks for picked epsilon value
    PLSSVM_ASSERT(std::numeric_limits<T>::epsilon() <= eps, "Chosen epsilon too small!: {} < {}", eps, std::numeric_limits<T>::epsilon());
    PLSSVM_ASSERT(eps < T{ 1.0 }, "Chosen epsilon too large!: {} >= 1.0", eps);

    if (val1 == val2) {
        SUCCEED();
    }

    const T diff = std::abs(val1 - val2);
    const T norm = std::min((std::abs(val1) + std::abs(val2)), std::numeric_limits<T>::max());

    if constexpr (expect) {
        EXPECT_LT(diff, std::max(std::numeric_limits<T>::min(), eps * norm)) << msg << " correct: " << val1 << " vs. actual: " << val2;
    } else {
        ASSERT_LT(diff, std::max(std::numeric_limits<T>::min(), eps * norm)) << msg << " correct: " << val1 << " vs. actual: " << val2;
    }
}

/**
 * @brief Compares the two vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first vector to compare
 * @param[in] val2 the second vector to compare
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_vector_near(const std::vector<T> &val1, const std::vector<T> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_near<T, expect>(val1[col], val2[col], fmt::format("values at [{}] are not equal enough: ", col));
    }
}
/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @tparam expect if `false` maps to `EXPECT_*`, else maps to `ASSERT_*`
 * @param[in] val1 the first 2D vector to compare
 * @param[in] val2 the second 2D vector to compare
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_2d_vector_near(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_near<T, expect>(val1[row][col], val2[row][col], fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
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
 * @details Other tests in the test case are aborted even if this test fails.
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
 * @details Other tests in the test case are aborted even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_EQ(val1, val2, msg) \
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
 * @details Other tests in the test case are aborted even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_EQ(val1, val2, msg) \
    detail::floating_point_2d_vector_eq<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

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
 * @details Other tests in the test case are aborted even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_NEAR(val1, val2, msg) \
    detail::floating_point_near<decltype(val1), false>(val1, val2, msg)

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
 * @details Other tests in the test case are aborted even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_VECTOR_NEAR(val1, val2, msg) \
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
 * @details Other tests in the test case are aborted even if this test fails.
 * @param[in] val1 the first value to compare (the actual value)
 * @param[in] val2 the second value to compare (the expected value)
 */
#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2, msg) \
    detail::floating_point_2d_vector_near<detail::get_value_type_t<plssvm::detail::remove_cvref_t<decltype(val1)>>, false>(val1, val2)

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

#endif  // PLSSVM_TESTS_CUSTOM_TEST_MACROS_HPP_