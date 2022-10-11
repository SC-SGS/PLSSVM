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

#ifndef PLSSVM_CUSTOM_TEST_MACROS_HPP_
#define PLSSVM_CUSTOM_TEST_MACROS_HPP_

#include "plssvm/detail/assert.hpp"   // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"  // plssvm::detail::always_false_v

#include "fmt/core.h"
#include "gtest/gtest.h"

#include <algorithm> // std::max
#include <cmath> // std::abs
#include <type_traits>  // std::is_same_v
#include <limits> // std::numeric_limits::{epsilon, max, min}

namespace impl {

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `EXPECT_FLOAT_EQ` and `EXPECT_DOUBLE_EQ`.
 * @tparam T the floating point type
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] msg an optional message
 */
template <typename T>
inline void expect_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_FLOAT_EQ(val1, val2) << msg;
    } else if constexpr (std::is_same_v<T, double>) {
        EXPECT_DOUBLE_EQ(val1, val2) << msg;
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "T must be either float or double!");
    }
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `ASSERT_FLOAT_EQ` and `ASSERT_DOUBLE_EQ`.
 * @tparam T the floating point type
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] msg an optional message
 */
template <typename T>
inline void assert_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
    if constexpr (std::is_same_v<T, float>) {
        ASSERT_FLOAT_EQ(val1, val2) << msg;
    } else if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(val1, val2) << msg;
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "T must be either float or double!");
    }
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
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
 * @param[in] val1 the first vector to compare
 * @param[in] val2 the second vector to compare
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_near(const std::vector<T> &val1, const std::vector<T> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type col = 0; col < val1.size(); ++col) {
        floating_point_near<T, expect>(val1[col], val2[col], fmt::format("values at [{}] are not equal enough: ", col));
    }
}
/**
 * @brief Compares the two 2D vectors of floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @param[in] val1 the first 2D vector to compare
 * @param[in] val2 the second 2D vector to compare
 * @param[in] msg an optional message
 */
template <typename T, bool expect>
inline void floating_point_near(const std::vector<std::vector<T>> &val1, const std::vector<std::vector<T>> &val2) {
    ASSERT_EQ(val1.size(), val2.size());
    for (typename std::vector<T>::size_type row = 0; row < val1.size(); ++row) {
        ASSERT_EQ(val1[row].size(), val2[row].size());
        for (typename std::vector<T>::size_type col = 0; col < val1[row].size(); ++col) {
            floating_point_near<T, expect>(val1[row][col], val2[row][col], fmt::format("values at [{}][{}] are not equal enough: ", row, col));
        }
    }
}
}  // namespace impl

#define EXPECT_FLOATING_POINT_EQ(val1, val2) \
    impl::expect_floating_point_eq(val1, val2)

#define ASSERT_FLOATING_POINT_EQ(val1, val2) \
    impl::assert_floating_point_eq(val1, val2)

#define EXPECT_FLOATING_POINT_NEAR(val1, val2) \
    impl::floating_point_near<decltype(val1), true>(val1, val2)

#define ASSERT_FLOATING_POINT_NEAR(val1, val2, msg) \
    impl::floating_point_near<decltype(val1), false>(val1, val2, msg)

#define EXPECT_FLOATING_POINT_VECTOR_NEAR(val1, val2) \
    impl::floating_point_near<typename plssvm::detail::remove_cvref_t<decltype(val1)>::value_type, true>(val1, val2)

#define ASSERT_FLOATING_POINT_VECTOR_NEAR(val1, val2, msg) \
    impl::floating_point_near<typename decltype(val1)::value_type, false>(val1, val2)

#define EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2) \
    impl::floating_point_near<typename (typename decltype(val1)::value_type)::value_type, true>(val1, val2)

#define ASSERT_FLOATING_POINT_2D_VECTOR_NEAR(val1, val2, msg) \
    impl::floating_point_near<typename (typename decltype(val1)::value_type)::value_type, false>(val1, val2)


#define EXPECT_THROW_WHAT(statement, expected_exception, msg)                                                \
    do {                                                                                                     \
        try {                                                                                                \
            statement;                                                                                       \
            FAIL() << "Expected " #expected_exception;                                                       \
        } catch (const expected_exception &e) {                                                              \
            EXPECT_EQ(std::string_view(e.what()), std::string_view(msg));                                    \
        } catch (...) {                                                                                      \
            FAIL() << "The expected exception type (" #expected_exception ") doesn't match the caught one!"; \
        }                                                                                                    \
    } while (false)

#endif  // PLSSVM_CUSTOM_TEST_MACROS_HPP_
