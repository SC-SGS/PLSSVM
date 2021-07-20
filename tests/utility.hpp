/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Utility functions for testing.
 */

#pragma once

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/utility.hpp"               // plssvm::detail::always_false_v
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, ASSERT_FLOAT_EQ, ASSERT_DOUBLE_EQ

#include <cstdlib>      // mkstemp
#include <filesystem>   // std::filesystem::temp_directory_path
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace util {

/**
 * @brief Create a unique temporary file in the temporary directory and return the file's name.
 * @return the name of the temporary file
 */
std::string create_temp_file() {
    std::string file = std::filesystem::temp_directory_path().string();
    file += "/tmpfile_XXXXXX";
    // create unique temporary file
    int fd = mkstemp(file.data());
    // immediately close file if possible
    if (fd >= 0) {
        close(fd);
    }
    return file;
}

/**
 * @brief Compares the two floating point values @p val1 and @p val2.
 * @details Wrapper around GoogleTest's `EXPECT_FLOAT_EQ` and `EXPECT_DOUBLE_EQ`.
 * @tparam T the floating point type
 * @param[in] val1 first value to compare
 * @param[in] val2 second value to compare
 * @param msg an optional message
 */
template <typename T>
void gtest_expect_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
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
 * @param msg an optional message
 */
template <typename T>
void gtest_assert_floating_point_eq(const T val1, const T val2, const std::string &msg = "") {
    if constexpr (std::is_same_v<T, float>) {
        ASSERT_FLOAT_EQ(val1, val2) << msg;
    } else if constexpr (std::is_same_v<T, double>) {
        ASSERT_DOUBLE_EQ(val1, val2) << msg;
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "T must be either float or double!");
    }
}

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