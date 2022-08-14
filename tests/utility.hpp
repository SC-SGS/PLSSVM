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
#include <filesystem>   // std::filesystem::temp_directory_path, std::filesystem::exists
#include <limits>       // std::numeric_limits
#include <sstream>      // std::ostringstream, std::istringstream
#include <string>       // std::string, std::to_string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v

namespace util {

/**
 * @brief Create a unique temporary file return the file's name.
 * @details On UNIX systems use `mkstemp` to create a unique file in the temporary directory. On non-UNIX system create
 *          a file in the current directory using a random `unsigned long long` number.
 * @return the name of the temporary file
 */
inline std::string create_temp_file() {
#ifdef __unix__
    std::string file = std::filesystem::temp_directory_path().string();
    file += "/tmpfile_XXXXXX";
    // create unique temporary file
    int fd = mkstemp(file.data());
    // immediately close file if possible
    if (fd >= 0) {
        close(fd);
    }
    return file;
#else
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned long long> dist;
    std::string file{ std::to_string(dist(gen)) };
    while (std::filesystem::exists(std::filesystem::current_path() / file)) {
        file = std::to_string(dist(gen));
    }
    return file;
#endif
}

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
 * @brief Check whether converting the enum @p e to a std::string results in the string representation @p str.
 * @tparam Enum the type of the enum to convert
 * @param[in] e the enum to convert to a std::string
 * @param[in] str the expected string representation of the enum @p e
 */
template <typename Enum>
inline void gtest_expect_enum_to_string_string_conversion(const Enum e, const std::string_view str) {
    std::ostringstream ss;
    ss << e;
    EXPECT_FALSE(ss.fail());
    EXPECT_EQ(ss.str(), str);
}
/**
 * @brief Check whether converting the string @p str to the enum type @p Enum yields @p e.
 * @tparam Enum the type of the enum
 * @param[in] str the string representation of @p e
 * @param[in] e the expected enum value
 */
template <typename Enum>
inline void gtest_expect_string_to_enum_conversion(const std::string &str, const Enum e) {
    std::istringstream ss{ str };
    Enum parsed{};
    ss >> parsed;
    EXPECT_FALSE(ss.fail());
    EXPECT_EQ(parsed, e);
}
/**
 * @brief Check whether converting the illegal string @p str to the enum type @p Enum results in setting the failbit in the stream object.
 * @tparam Enum the type of the enum
 * @param[in] str the illegal string representation of an enum value of type @p Enum
 */
template <typename Enum>
inline void gtest_expect_string_to_enum_conversion(const std::string &str) {
    std::istringstream ss{ str };
    Enum parsed{};
    ss >> parsed;
    EXPECT_TRUE(ss.fail());
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


/*
 * Convert the parameter to a std::string using a std::ostringstream.
 */
template <typename T>
inline std::string convert_to_string(const T &param) {
    std::ostringstream os;
    os << param;
    // test if output was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(os.fail());
    }();
    return os.str();
}
/*
 * Convert the std::string to a value of type T using a std::istringstream.
 */
template <typename T>
inline T convert_from_string(const std::string &str) {
    std::istringstream is{ str };
    T param{};
    is >> param;
    // test if input was successful
    [&]() {
        // need immediate invoked lambda because of void return in ASSERT_FALSE
        ASSERT_FALSE(is.fail());
    }();
    return param;
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
#define EXPECT_THROW_WHAT(statement, expected_exception, msg)                     \
    do {                                                                          \
        try {                                                                     \
            statement;                                                            \
            FAIL() << "Expected " #expected_exception;                            \
        } catch (const expected_exception &e) {                                   \
            EXPECT_EQ(std::string_view(e.what()), std::string_view(msg));         \
        } catch (...) {                                                           \
            FAIL() << "Expected " #expected_exception " with message: " << (msg); \
        }                                                                         \
    } while (false)

namespace google_test {
/**
 * @brief Save the data type and kernel function in a single struct used by GoogleTest's TYPED_TEST_SUITE.
 * @tparam T the type of the data
 * @tparam Kernel the kernel function to use
 */
template <typename T, plssvm::kernel_type Kernel>
struct parameter_definition {
    using real_type = T;
    static constexpr plssvm::kernel_type kernel = Kernel;
};

/**
 * @brief Class used to pretty print a `util::google_test::parameter_definition` in a test case name.
 */
class parameter_definition_to_name {
  public:
    /**
     * @brief Convert a `util::google_test::parameter_definition` to a string representation.
     * @tparam T `util::google_test::parameter_definition`
     * @return the string representation
     */
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}.{}", plssvm::detail::arithmetic_type_name<typename T::real_type>(), T::kernel);
    }
};

}  // namespace google_test

}  // namespace util