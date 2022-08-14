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

#include "../utility.hpp"  // EXPECT_THROW_WHAT
#include "utility.hpp"     // util::{exception_type_name, exception_definition, exception_definition_to_name}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_THAT, EXPECT_TRUE, ASSERT_EQ, ::testing::Test, ::testing::Types

#include <regex>        // std::regex, std::regex_matcher
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

// helper function returning an exception used to be able to name the source location function
template <typename Exception>
Exception dummy(const std::string &msg) {
    return Exception{ msg };
}

// enumerate all custom exception types; ATTENTION: don't forget to also specialize the PLSSVM_CREATE_EXCEPTION_TYPE_NAME macro if a new exception type is added
using exception_types = ::testing::Types<plssvm::exception, plssvm::invalid_parameter_exception, plssvm::file_not_found_exception, plssvm::invalid_file_format_exception,
                                         plssvm::unsupported_backend_exception, plssvm::unsupported_kernel_type_exception, plssvm::gpu_device_ptr_exception>;

template <typename T>
class Exceptions : public ::testing::Test {};
TYPED_TEST_SUITE(Exceptions, exception_types);

// check whether throwing exceptions works as intended
TYPED_TEST(Exceptions, throwing_excpetion) {
    using exception_type = TypeParam;
    //    EXPECT_THROW_WHAT(throw_exception<exception_type>("exception message"), exception_type, "exception message");
    EXPECT_THROW_WHAT(throw exception_type{ "exception message" }, exception_type, "exception message");
}

// check whether the source location information are populated correctly
TYPED_TEST(Exceptions, exception_source_location) {
    const auto e = dummy<TypeParam>("exception message");

    EXPECT_EQ(e.loc().file_name(), __FILE__);
    EXPECT_THAT(e.loc().function_name(), ::testing::HasSubstr("dummy"));
    EXPECT_EQ(e.loc().line(), 28);   // attention: hardcoded line!
    EXPECT_EQ(e.loc().column(), 0);  // attention: always 0!
}

// check whether what message including the source location information is assembled correctly
TYPED_TEST(Exceptions, exception_what_with_source_location) {
    using exception_type = TypeParam;
    const auto e = dummy<exception_type>("exception message");

    // get exception message with source location information split into a vector of separate lines
    const std::string what = e.what_with_loc();
    const std::vector<std::string_view> what_lines = plssvm::detail::split(what, '\n');

    // create vector containing correct regex
    std::vector<std::string> regex_patterns;
    regex_patterns.emplace_back("exception message");
    regex_patterns.emplace_back(fmt::format("{} thrown:", util::exception_type_name<exception_type>()));
    regex_patterns.emplace_back("  in file      " __FILE__);
    regex_patterns.emplace_back("  in function  .*dummy.*");
    regex_patterns.emplace_back("  @ line       28");

    // number of lines and what message and regex matchers must be equal
    ASSERT_EQ(what_lines.size(), regex_patterns.size());

    // check if what message matches the regex
    for (std::vector<std::string>::size_type i = 0; i < regex_patterns.size(); ++i) {
        std::regex reg(regex_patterns[i], std::regex::extended);
        EXPECT_TRUE(std::regex_match(what_lines[i].data(), what_lines[i].data() + what_lines[i].size(), reg)) << fmt::format(R"(line {}: "{}" doesn't match regex pattern: "{}")", i, what_lines[i], regex_patterns[i]);
    }
}