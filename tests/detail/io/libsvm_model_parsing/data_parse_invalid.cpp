/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing an invalid LIBSVM model file data section.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/constants.hpp"                    // plssvm::real_type
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/io/file_reader.hpp"        // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"        // plssvm::invalid_file_format_exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::label_type_to_name
#include "types_to_test.hpp"       // util::label_type_gtest

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, ::testing::Test

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::ignore
#include <vector>   // std::vector

template <typename T>
class LIBSVMModelDataParseInvalid : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMModelDataParseInvalid, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelDataParseInvalid, zero_based_features) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/zero_based_features.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, empty_data) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/empty_data.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no data points are given!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, too_few_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed at least 1 alpha values, but fewer (0) were provided!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, too_many_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_many_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed at most 2 alpha values, but more (3) were provided!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, feature_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/feature_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert 'a-1.1178275006e+00' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TYPED_TEST(LIBSVMModelDataParseInvalid, index_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/index_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' !2' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, invalid_colon_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/invalid_colon_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed at least 1 alpha values, but fewer (0) were provided!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, invalid_colon_in_the_middle) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/invalid_colon_in_the_middle.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, missing_feature_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_feature_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TYPED_TEST(LIBSVMModelDataParseInvalid, missing_index_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/missing_index_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, non_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/non_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 3 is smaller or equal than 3!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, non_strictly_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/non_strictly_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 2 is smaller or equal than 3!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, oaa_and_oao) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/oaa_and_oao.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't distinguish between OAA and OAO in the given model file!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, too_many_num_sv_per_class) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_many_num_sv_per_class.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 2, 2, 1, 1 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed at least 3 alpha values, but fewer (2) were provided!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, too_few_sv_according_to_header) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_few_sv_according_to_header.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Found 5 support vectors, but it should be 6!");
}
TYPED_TEST(LIBSVMModelDataParseInvalid, too_many_sv_according_to_header) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/invalid/too_many_sv_according_to_header.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 8)),
                      plssvm::invalid_file_format_exception,
                      "Found 7 support vectors, but it should be 6!");
}

template <typename T>
class LIBSVMModelDataParseInvalidDeathTest : public LIBSVMModelDataParseInvalid<T> {};
TYPED_TEST_SUITE(LIBSVMModelDataParseInvalidDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelDataParseInvalidDeathTest, invalid_file_reader) {
    // open file_reader without associating it to a file
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 0)),
                 "The file_reader is currently not associated with a file!");
}
TYPED_TEST(LIBSVMModelDataParseInvalidDeathTest, too_few_num_sv_per_class) {
    // parse LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // try to skip more lines than are present in the data file
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3 }, 0)),
                 "At least two classes must be present!");
}
TYPED_TEST(LIBSVMModelDataParseInvalidDeathTest, skip_too_many_lines) {
    // parse LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/6x4_linear.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // try to skip more lines than are present in the data file
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data(reader, std::vector<std::size_t>{ 3, 3 }, 15)),
                 "Tried to skipp 15 lines, but only 14 are present!");
}