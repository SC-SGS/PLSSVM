/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for testing.
 */

#ifndef PLSSVM_TESTS_UTILITY_HPP_
#define PLSSVM_TESTS_UTILITY_HPP_
#pragma once

#include "plssvm/csvm_factory.hpp"                 // plssvm::make_csvm
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"               // plssvm::detail::always_false_v
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // make custom type formattable using operator>> overload
#include "gtest/gtest.h"  // EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, ASSERT_FLOAT_EQ, ASSERT_DOUBLE_EQ, EXPECT_EQ, EXPECT_NE, EXPECT_FALSE, EXPECT_TRUE, EXPECT_LT, SUCCESS, FAIL

#ifdef __unix__
    #include <cstdlib>  // mkstemp
#else
    #include <random>  // std::random_device, std::mt19937, std::uniform_int_distribution
#endif

#include <algorithm>    // std::min, std::max
#include <cmath>        // std::abs
#include <filesystem>   // std::filesystem::temp_directory_path, std::filesystem::exists, std::filesystem::remove
#include <iostream>     // std::cout
#include <limits>       // std::numeric_limits
#include <random>       // std::random_device, std::mt19937, std::uniform_real_distribution
#include <sstream>      // std::ostringstream, std::istringstream, std::stringstream
#include <streambuf>    // std::streambuf
#include <string>       // std::string, std::to_string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v

namespace util {

/**
 * @brief Class used to redirect the standard output inside test cases.
 */
class redirect_output {
  public:
    /**
     * @brief Redirect the output and store the original standard output location.
     */
    redirect_output() {
        // capture std::cout
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }
    /**
     * @brief Restore the original standard output location.
     */
    ~redirect_output() {
        // end capturing std::cout
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }

  private:
    std::stringstream buffer_{};
    std::streambuf *sbuf_{ nullptr };
};

/**
 * @brief A class encapsulating unique temporary file's name.
 * @details On UNIX systems use `mkstemp` to create a unique file in the temporary directory. On non-UNIX system create
 *          a file in the current directory using a random `unsigned long long` number.
 */
class temporary_file {
  public:
    /**
     * @brief Create a new unique temporary file.
     */
    temporary_file() {
#ifdef __unix__
        filename = std::filesystem::temp_directory_path().string();
        filename += "/tmpfile_XXXXXX";
        // create unique temporary file
        const int file_descriptor = mkstemp(filename.data());
        // immediately close file if possible
        if (file_descriptor >= 0) {
            close(file_descriptor);
        }
#else
        std::random_device device;
        std::mt19937 gen(device());
        std::uniform_int_distribution<unsigned long long> dist;
        filename{ std::to_string(dist(gen)) };
        while (std::filesystem::exists(std::filesystem::temp_directory_path() / filename)) {
            filename = std::to_string(dist(gen));
        }
        // create file
        std::ofstream{ std::filesystem::temp_directory_path() / filename) };
#endif
    }
    /**
     * @brief Remove the temporary file if it exists.
     */
    ~temporary_file() {
        std::filesystem::remove(filename);
    }

    std::string filename;
};

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `EXPECT_FLOAT_EQ` and `EXPECT_DOUBLE_EQ`.
 * @tparam T the floating point type
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] msg an optional message
 */
template <typename T>
inline void gtest_expect_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
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
inline void gtest_assert_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
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
template <typename T>
inline void gtest_assert_floating_point_near(const T val1, const T val2, const T scale, const std::string &msg = "") {
    // based on: https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison

    // set epsilon
    const T eps = 128 * scale * std::numeric_limits<T>::epsilon();

    // sanity checks for picked epsilon value
    PLSSVM_ASSERT(std::numeric_limits<T>::epsilon() <= eps, "Chosen epsilon too small!: {} < {}", eps, std::numeric_limits<T>::epsilon());
    PLSSVM_ASSERT(eps < T{ 1.0 }, "Chosen epsilon too large!: {} >= 1.0", eps);

    if (val1 == val2) {
        SUCCEED();
    }

    const T diff = std::abs(val1 - val2);
    const T norm = std::min((std::abs(val1) + std::abs(val2)), std::numeric_limits<T>::max());

    EXPECT_LT(diff, std::max(std::numeric_limits<T>::min(), eps * norm)) << msg << " correct: " << val1 << " vs. actual: " << val2;
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2 using a mixture of relative and absolute mode.
 * @tparam T the floating point type
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param[in] msg an optional message
 */
template <typename T>
inline void gtest_assert_floating_point_near(const T val1, const T val2, const std::string &msg = "") {
    gtest_assert_floating_point_near(val1, val2, T{ 1 }, msg);
}

/**
 * @brief Check whether the C-SVM returned by a call to `plssvm::make_csvm` with the parameters @p params returns a C-SVM of type @p Derived
 * @tparam Derived the expected C-SVM type, based on the backend_type of @þ params
 * @tparam T the type of the data
 * @param[in] params the parameter to initialize the C-SVM with
 */
template <template <typename> typename Derived, typename T>
inline void gtest_expect_correct_csvm_factory(const plssvm::parameter<T> &params) {
    // create csvm
    auto csvm = plssvm::make_csvm(params);
    // check if correct csvm has been instantiated
    auto *res = dynamic_cast<Derived<T> *>(csvm.get());
    EXPECT_NE(res, nullptr);
}

/**
 * @brief Convert the parameter @p value to a std::string using a std::ostringstream.
 */
template <typename T>
inline std::string convert_to_string(const T &value) {
    std::ostringstream output;
    output << value;
    // test if output was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(output.fail());
    }();
    return output.str();
}
/**
 * @brief Convert the std::string @p str to a value of type T using a std::istringstream.
 */
template <typename T>
inline T convert_from_string(const std::string &str) {
    std::istringstream input{ str };
    T value{};
    input >> value;
    // test if input was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(input.fail());
    }();
    return value;
}

/**
 * @brief Generate vector of @p size filled with random values in the range [@p lower, @p upper].
 * @tparam T the type of the elements in the vector
 * @param[in] size the size of the vector
 * @param[in] lower the lower bound of the random values in the vector
 * @param[in] upper the upper bound of the random values in the vector
 * @return the randomly generated vector (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::vector<T> generate_random_vector(const std::size_t size, const T lower = T{ -1.0 }, const T upper = T{ 1.0 }) {
    std::vector<T> vec(size);

    // fill vectors with random values
    static std::random_device device;
    static std::mt19937 gen(device());
    std::uniform_real_distribution<T> dist(lower, upper);
    std::generate(vec.begin(), vec.end(), [&]() { return dist(gen); });

    return vec;
}

/**
 * @brief Defines a macro like
 *        <a href="https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/docs/advanced.md">googletest</a>'s
 *        `EXPECT_THROW`, but also allows to test for the correct exception's
 *        [`what()`](https://en.cppreference.com/w/cpp/error/exception/what) message.
 *
 * @param[in] statement the statement which should throw (a specific exception)
 * @param[in] expected_exception the type of the exception which should get thrown
 * @param[in] msg the expected exception's [`what()`](https://en.cppreference.com/w/cpp/error/exception/what) message
 */
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

}  // namespace util

#endif  // PLSSVM_TESTS_UTILITY_HPP_