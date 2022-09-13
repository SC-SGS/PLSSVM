/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the file reader implementation that uses memory-mapped IO if possible.
 */

#include "plssvm/detail/io/file_reader.hpp"

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "utility.hpp"                // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <vector>  // std::vector

class FileReaderNumlines : public ::testing::TestWithParam<std::pair<std::string, typename std::string_view::size_type>> {};

TEST_P(FileReaderNumlines, num_lines) {
    const auto& [filename, count] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines();

    // check if the read number of lines is correct
    EXPECT_EQ(reader.num_lines(), count);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(FileReader, FileReaderNumlines, ::testing::Values(
                         std::make_pair(PLSSVM_TEST_PATH "/data/arff/5x4.arff", 17 - 3),
                         std::make_pair(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", 5),
                         std::make_pair(PLSSVM_TEST_PATH "/data/libsvm/500x200.libsvm", 501 - 1),
                         std::make_pair(PLSSVM_TEST_PATH "/data/empty.txt", 0)));
// clang-format on