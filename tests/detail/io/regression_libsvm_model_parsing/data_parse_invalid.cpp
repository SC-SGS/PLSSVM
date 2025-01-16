/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing an invalid LIBSVM regression model file data section.
 */

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/detail/arithmetic_type_name.hpp"                // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/exceptions/exceptions.hpp"                      // plssvm::invalid_file_format_exception

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::label_type_to_name
#include "tests/types_to_test.hpp"       // util::regression_label_type_gtest

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, ::testing::Test

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::ignore
#include <vector>   // std::vector

template <typename T>
class LIBSVMRegressionModelDataParseInvalid : public ::testing::Test { };

TYPED_TEST_SUITE(LIBSVMRegressionModelDataParseInvalid, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, zero_based_features) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/zero_based_features.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, empty_data) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/empty_data.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no data points are given!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, missing_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed exactly one alpha value, but none were provided!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, too_many_alpha_values) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/too_many_alpha_values.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed exactly one alpha value, but more were provided!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, feature_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/feature_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert 'a-1.1178275006e+00' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, index_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/index_with_alpha_char_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' !2' to a value of type unsigned long!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, invalid_colon_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/invalid_colon_at_the_beginning.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: needed exactly one alpha value, but none were provided!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, invalid_colon_in_the_middle) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/invalid_colon_in_the_middle.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, missing_feature_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_feature_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, missing_index_value) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/missing_index_value.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, non_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/non_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 3 is smaller or equal than 3!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalid, non_strictly_increasing_indices) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/invalid/non_strictly_increasing_indices.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 6)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 2 is smaller or equal than 3!");
}

template <typename T>
class LIBSVMRegressionModelDataParseInvalidDeathTest : public LIBSVMRegressionModelDataParseInvalid<T> { };

TYPED_TEST_SUITE(LIBSVMRegressionModelDataParseInvalidDeathTest, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelDataParseInvalidDeathTest, invalid_file_reader) {
    // open file_reader without associating it to a file
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 0)),
                 "The file_reader is currently not associated with a file!");
}

TYPED_TEST(LIBSVMRegressionModelDataParseInvalidDeathTest, skip_too_many_lines) {
    // parse LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/model/regression/6x4_linear.libsvm.model";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // try to skip more lines than are present in the data file
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_model_data_regression(reader, 15)),
                 "Tried to skipp 15 lines, but only 12 are present!");
}
