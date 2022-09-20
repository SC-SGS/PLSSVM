/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to parsing the data points from a ARFF file.
 */

#include "plssvm/detail/io/arff_parsing.hpp"

#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "../../utility.hpp"  // util::create_temp_file, EXPECT_THROW_WHAT

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_P, TYPED_TEST, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, GTEST_FAIL
                                   // ::testing::{Test, Types, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <string>       // std::string
#include <tuple>        // std::ignore
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

// struct for the used type combinations
template <typename T, typename U>
struct type_combinations {
    using real_type = T;
    using label_type = U;
};

// the floating point and label types combinations to test
using type_combinations_types = ::testing::Types<
    type_combinations<float, int>,
    type_combinations<float, std::string>,
    type_combinations<double, int>,
    type_combinations<double, std::string>>;

class ARFFParseHeader : public ::testing::Test {};
class ARFFParseHeaderValid : public ::testing::TestWithParam<std::tuple<std::string, std::size_t, std::size_t, bool>> {};
TEST_P(ARFFParseHeaderValid, header) {
    const auto &[filename_part, num_features, header_skip, has_label] = GetParam();

    // parse the ARFF file
    const std::string filename = fmt::format("{}{}", PLSSVM_TEST_PATH, filename_part);
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    const auto& [parsed_num_features, parsed_header_skip, parsed_has_label] = plssvm::detail::io::parse_arff_header(reader);

    // check for correctness
    EXPECT_EQ(parsed_num_features, num_features);
    EXPECT_EQ(parsed_header_skip, header_skip);
    EXPECT_EQ(parsed_has_label, has_label);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ARFFParse, ARFFParseHeaderValid, ::testing::Values(
                                                     std::make_tuple("/data/arff/5x4_int.arff", 4, 7, true),
                                                     std::make_tuple("/data/arff/5x4_string.arff", 4, 7, true),
                                                     std::make_tuple("/data/arff/5x4_sparse_int.arff", 4, 7, true),
                                                     std::make_tuple("/data/arff/5x4_sparse_string.arff", 4, 7, true),
                                                     std::make_tuple("/data/arff/3x2_without_label.arff", 2, 4, false)));
// clang-format on

TEST(ARFFParseHeader, class_unquoted_nominal_attribute) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_unquoted_nominal_attribute.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "The \"@ATTRIBUTE class    -1,1\" nominal attribute must be enclosed with {}!");
}
TEST(ARFFParseHeader, class_with_wrong_label) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_with_wrong_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "May not use the combination of the reserved name \"class\" and attribute type NUMERIC!");
}
TEST(ARFFParseHeader, class_without_label) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_without_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "The \"@ATTRIBUTE class\" field must contain class labels!");
}
TEST(ARFFParseHeader, multiple_classes) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/multiple_classes.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "A nominal attribute with the name CLASS may only be provided once!");
}
TEST(ARFFParseHeader, no_features) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/no_features.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "Can't parse file: no feature ATTRIBUTES are defined!");
}
TEST(ARFFParseHeader, no_data_attribute) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/no_data_attribute.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "Can't parse file: @DATA is missing!");
}
TEST(ARFFParseHeader, nominal_attribute_with_wrong_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/nominal_attribute_with_wrong_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "Read an invalid header entry: \"@ATTRIBUTE foo    {-1,1}\"!");
}
TEST(ARFFParseHeader, numeric_unquoted) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/numeric_unquoted.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "A \"@ATTRIBUTE second entry   numeric\" name that contains a whitespace must be quoted!");
}
TEST(ARFFParseHeader, numeric_without_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/numeric_without_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "The \"@ATTRIBUTE   numeric\" field must contain a name!");
}
TEST(ARFFParseHeader, relation_not_at_beginning) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_not_at_beginning.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "The @RELATION attribute must be set before any other @ATTRIBUTE!");
}
TEST(ARFFParseHeader, relation_unquoted) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_unquoted.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "A \"@RELATION  name with whitespaces\" name that contains a whitespace must be quoted!");
}
TEST(ARFFParseHeader, relation_without_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_without_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "The \"@RELATION\" field must contain a name!");
}
TEST(ARFFParseHeader, wrong_line) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/wrong_line.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "Read an invalid header entry: \"@THIS IS NOT A CORRECT LINE!\"!");
}
TEST(ARFFParseHeader, empty) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header(reader)), plssvm::invalid_file_format_exception, "Can't parse file: no feature ATTRIBUTES are defined!");
}