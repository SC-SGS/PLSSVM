/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the custom exception classes.
 */

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::{*_exception}

#include "../custom_test_macros.hpp"         // EXPECT_THROW_WHAT
#include "../naming.hpp"                     // naming::exception_types_to_name
#include "utility.hpp"                       // util::exception_type_name

#include "fmt/core.h"                        // fmt::format
#include "gmock/gmock.h"                     // EXPECT_THAT, ::testing::{HasSubstr, ContainsRegex}
#include "gtest/gtest.h"                     // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, ASSERT_EQ, ::testing::Test, ::testing::Types

#include <string>                            // std::string
#include <string_view>                       // std::string_view
#include <vector>                            // std::vector

// helper function returning an exception used to be able to name the source location function
template <typename Exception>
Exception dummy(const std::string &msg) {
    return Exception{ msg };
}

// clang-format off
// enumerate all custom exception types; ATTENTION: don't forget to also specialize the PLSSVM_CREATE_EXCEPTION_TYPE_NAME macro if a new exception type is added
using exception_types = ::testing::Types<plssvm::exception, plssvm::invalid_parameter_exception, plssvm::file_reader_exception,
                                         plssvm::data_set_exception, plssvm::file_not_found_exception, plssvm::invalid_file_format_exception,
                                         plssvm::unsupported_backend_exception, plssvm::unsupported_kernel_type_exception, plssvm::gpu_device_ptr_exception>;
// clang-format on

template <typename T>
class Exceptions : public ::testing::Test {};
TYPED_TEST_SUITE(Exceptions, exception_types, naming::exception_types_to_name);

// check whether throwing exceptions works as intended
TYPED_TEST(Exceptions, throwing_excpetion) {
    using exception_type = TypeParam;
    // throw the specified exception
    const auto dummy = []() { throw exception_type{ "exception message" }; };
    EXPECT_THROW_WHAT(dummy(), exception_type, "exception message");
}

// check whether the source location information are populated correctly
TYPED_TEST(Exceptions, exception_source_location) {
    const auto exc = dummy<TypeParam>("exception message");

    EXPECT_EQ(exc.loc().file_name(), __FILE__);
    EXPECT_THAT(exc.loc().function_name(), ::testing::HasSubstr("dummy"));
    EXPECT_EQ(exc.loc().line(), 28);   // attention: hardcoded line!
    EXPECT_EQ(exc.loc().column(), 0);  // attention: always 0!
}

// check whether what message including the source location information is assembled correctly
TYPED_TEST(Exceptions, exception_what_with_source_location) {
    using exception_type = TypeParam;
    const auto exc = dummy<exception_type>("exception message");

    // get exception message with source location information split into a vector of separate lines
    const std::string what = exc.what_with_loc();
    const std::vector<std::string_view> what_lines = plssvm::detail::split(what, '\n');

    // check the number of lines in the "what" message
    ASSERT_EQ(what_lines.size(), 5);

    // check the "what" message content
    EXPECT_EQ(what_lines[0], "exception message");
    EXPECT_EQ(what_lines[1], fmt::format("{} thrown:", util::exception_type_name<exception_type>()));
    EXPECT_EQ(what_lines[2], "  in file      " __FILE__);
    EXPECT_THAT(std::string{ what_lines[3] }, ::testing::ContainsRegex("  in function  .*dummy.*"));
    EXPECT_EQ(what_lines[4], "  @ line       28");
}