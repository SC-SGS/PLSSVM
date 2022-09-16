/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to parsing the data points from a LIBSVM file.
 */

#include "plssvm/detail/io/libsvm_parsing.hpp"

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

class LIBSVMParseNumFeatures : public ::testing::TestWithParam<std::pair<std::string, std::size_t>> {};
TEST_P(LIBSVMParseNumFeatures, num_features) {
    const auto &[filename_part, num_features] = GetParam();

    // parse the LIBSVM file
    const std::string filename = fmt::format("{}{}", PLSSVM_TEST_PATH, filename_part);
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_EQ((plssvm::detail::io::parse_libsvm_num_features(reader.lines())), num_features);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(LIBSVMParse, LIBSVMParseNumFeatures, ::testing::Values(
                                                      std::make_pair("/data/libsvm/5x4_int.libsvm", 4),
                                                      std::make_pair("/data/libsvm/5x4_sparse_string.libsvm", 4),
                                                      std::make_pair("/data/libsvm/3x2_without_label.libsvm", 2),
                                                      std::make_pair("/data/libsvm/500x200.libsvm", 200),
                                                      std::make_pair("/data/empty.txt", 0)));
// clang-format on

TEST(LIBSVMParseNumFeatures, index_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/index_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_num_features(reader.lines())), plssvm::invalid_file_format_exception, "Can't convert ' !2' to a value of type unsigned long!");
}

template <typename T>
class LIBSVMParseDense : public ::testing::Test {
  protected:
    void SetUp() override {
        // fill label vector based on the label_type
        if constexpr (std::is_same_v<label_type, int>) {
            correct_label = std::vector<int>{ 1, 1, -1, -1, -1 };
            filename = PLSSVM_TEST_PATH "/data/libsvm/5x4_int.libsvm";
        } else if constexpr (std::is_same_v<label_type, std::string>) {
            correct_label = std::vector<std::string>{ "cat", "cat", "dog", "dog", "dog" };
            filename = PLSSVM_TEST_PATH "/data/libsvm/5x4_string.libsvm";
        }
    }

    using real_type = typename T::real_type;
    using label_type = typename T::label_type;

    std::string filename{};

    const std::vector<std::vector<real_type>> correct_data{
        { real_type{ -1.117827500607882 }, real_type{ -2.9087188881250993 }, real_type{ 0.66638344270039144 }, real_type{ 1.0978832703949288 } },
        { real_type{ -0.5282118298909262 }, real_type{ -0.335880984968183973 }, real_type{ 0.51687296029754564 }, real_type{ 0.54604461446026 } },
        { real_type{ 0.57650218263054642 }, real_type{ 1.01405596624706053 }, real_type{ 0.13009428079760464 }, real_type{ 0.7261913886869387 } },
        { real_type{ -0.20981208921241892 }, real_type{ 0.60276937379453293 }, real_type{ -0.13086851759108944 }, real_type{ 0.10805254527169827 } },
        { real_type{ 1.88494043717792 }, real_type{ 1.00518564317278263 }, real_type{ 0.298499933047586044 }, real_type{ 1.6464627048813514 } }
    };
    std::vector<label_type> correct_label{};
};
TYPED_TEST_SUITE(LIBSVMParseDense, type_combinations_types);

template <typename T>
class LIBSVMParseSparse : public ::testing::Test {
  protected:
    void SetUp() override {
        // fill label vector based on the label_type
        if constexpr (std::is_same_v<label_type, int>) {
            correct_label = std::vector<int>{ 1, 1, -1, -1, -1 };
            filename = PLSSVM_TEST_PATH "/data/libsvm/5x4_sparse_int.libsvm";
        } else if constexpr (std::is_same_v<label_type, std::string>) {
            correct_label = std::vector<std::string>{ "cat", "cat", "dog", "dog", "dog" };
            filename = PLSSVM_TEST_PATH "/data/libsvm/5x4_sparse_string.libsvm";
        }
    }

    using real_type = typename T::real_type;
    using label_type = typename T::label_type;

    std::string filename{};
    std::string filename_without_label{};

    const std::vector<std::vector<real_type>> correct_data{
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.51687296029754564 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 1.01405596624706053 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.60276937379453293 }, real_type{ 0.0 }, real_type{ -0.13086851759108944 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.298499933047586044 } }
    };
    std::vector<label_type> correct_label{};
};
TYPED_TEST_SUITE(LIBSVMParseSparse, type_combinations_types);

template <typename T>
class LIBSVMParse : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMParse, type_combinations_types);

TYPED_TEST(LIBSVMParseDense, read) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 5);
    ASSERT_EQ(num_features, 4);

    EXPECT_EQ(data, this->correct_data);

    EXPECT_EQ(label, this->correct_label);
}
TYPED_TEST(LIBSVMParseDense, read_skip_lines) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    // skip half of all lines
    const std::size_t skipped = 3;
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader, skipped);

    // check for correct sizes
    ASSERT_EQ(num_data_points, reader.num_lines() - skipped);
    ASSERT_EQ(num_features, 4);

    for (std::size_t i = 0; i < num_data_points; ++i) {
        EXPECT_EQ(data[i], this->correct_data[skipped + i]);
    }

    for (std::size_t i = 0; i < num_data_points; ++i) {
        EXPECT_EQ(label[i], this->correct_label[skipped + i]);
    }
}

TYPED_TEST(LIBSVMParseSparse, read) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 5);
    ASSERT_EQ(num_features, 4);

    EXPECT_EQ(data, this->correct_data);

    EXPECT_EQ(label, this->correct_label);
}
TYPED_TEST(LIBSVMParseSparse, read_skip_lines) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    // skip half of all lines
    const std::size_t skipped = 3;
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader, skipped);

    // check for correct sizes
    ASSERT_EQ(num_data_points, reader.num_lines() - skipped);
    ASSERT_EQ(num_features, 4);

    for (std::size_t i = 0; i < num_data_points; ++i) {
        EXPECT_EQ(data[i], this->correct_data[i + skipped]);
    }

    for (std::size_t i = 0; i < num_data_points; ++i) {
        EXPECT_EQ(label[i], this->correct_label[i + skipped]);
    }
}

TYPED_TEST(LIBSVMParse, read_without_label) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 3);
    ASSERT_EQ(num_features, 2);

    const std::vector<std::vector<current_real_type>> correct_data{
        { current_real_type{ 1.5 }, current_real_type{ -2.9 } },
        { current_real_type{ 0.0 }, current_real_type{ -0.3 } },
        { current_real_type{ 5.5 }, current_real_type{ 0.0 } }
    };
    EXPECT_EQ(data, correct_data);

    EXPECT_TRUE(label.empty());
}

TYPED_TEST(LIBSVMParse, zero_based_features) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/zero_based_features.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}
TYPED_TEST(LIBSVMParse, arff_file) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/5x4.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception);
}
TYPED_TEST(LIBSVMParse, empty) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Can't parse file: no data points are given!");
}
TYPED_TEST(LIBSVMParse, feature_with_alpha_char_at_the_beginning) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/feature_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, fmt::format("Can't convert 'a-1.11' to a value of type {}!", plssvm::detail::arithmetic_type_name<current_real_type>()));
}
TYPED_TEST(LIBSVMParse, index_with_alpha_char_at_the_beginning) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/index_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Can't convert ' !2' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, invalid_colon_at_the_beginning) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/invalid_colon_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Can't convert '' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, invalid_colon_in_the_middle) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/invalid_colon_in_the_middle.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Can't convert ' :2' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, missing_feature_value) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/missing_feature_value.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<current_real_type>()));
}
TYPED_TEST(LIBSVMParse, missing_index_value) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/missing_index_value.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Can't convert ' ' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, inconsistent_label_specification) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/inconsistent_label_specification.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "Inconsistent label specification found (some data points are labeled, others are not)!");
}
TYPED_TEST(LIBSVMParse, non_increasing_indices) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/non_increasing_indices.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "The features indices must be strictly increasing, but 3 is smaller or equal than 3!");
}
TYPED_TEST(LIBSVMParse, non_strictly_increasing_indices) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/non_strictly_increasing_indices.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception, "The features indices must be strictly increasing, but 2 is smaller or equal than 3!");
}

template <typename T>
class LIBSVMParseDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMParseDeathTest, type_combinations_types);

TYPED_TEST(LIBSVMParseDeathTest, invalid_file_reader) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // open file_reader without associating it to a file
    plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), "The file_reader is currently not associated with a file!");
}
TYPED_TEST(LIBSVMParseDeathTest, skip_too_many_lines) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/5x4_int.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    // try to skip more lines than are present in the data file
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader, 6)), "Tried to skipp 6 lines, but only 5 are present!");
}

template <typename T>
class LIBSVMWriteBase : public ::testing::Test {
  protected:
    void SetUp() override {
        // create a temporary file containing the scaling factors
        filename = util::create_temp_file();
    }
    void TearDown() override {
        // remove the temporary file at the end
        std::filesystem::remove(filename);
    }

    std::string filename;
};

template <typename T>
class LIBSVMWrite : public LIBSVMWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMWrite, type_combinations_types);

template <typename T>
class LIBSVMWriteDeathTest : public LIBSVMWriteBase<T> {};
TYPED_TEST_SUITE(LIBSVMWriteDeathTest, type_combinations_types);

TYPED_TEST(LIBSVMWrite, write_dense_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ -1, 1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "cat", "dog", "cat" };
    }

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.size());
    // check the lines
    for (std::size_t i = 0; i < data.size(); ++i) {
        const std::string line = fmt::format("{} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", label[i], data[i][0], data[i][1], data[i][2]);
        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            GTEST_FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}
TYPED_TEST(LIBSVMWrite, write_dense_without_label) {
    using real_type = typename TypeParam::real_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.1 }, real_type{ 1.2 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 2.2 }, real_type{ 2.3 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 3.3 } }
    };

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.size());
    // check the lines
    for (std::size_t i = 0; i < data.size(); ++i) {
        const std::string line = fmt::format("1:{:.10e} 2:{:.10e} 3:{:.10e} ", data[i][0], data[i][1], data[i][2]);
        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            GTEST_FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(LIBSVMWrite, write_sparse_with_label) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 0.0 } }
    };
    std::vector<label_type> label{};
    if constexpr (std::is_same_v<label_type, int>) {
        label = std::vector<int>{ -1, 1, -1 };
    } else if constexpr (std::is_same_v<label_type, std::string>) {
        label = std::vector<std::string>{ "cat", "dog", "cat" };
    }

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.size());
    // check the lines
    for (std::size_t i = 0; i < data.size(); ++i) {
        // assemble correct line
        std::string line = fmt::format("{} ", label[i]);
        for (std::size_t j = 0; j < data[i].size(); ++j) {
            if (data[i][j] != real_type{ 0.0 }) {
                line += fmt::format("{}:{:.10e} ", j + 1, data[i][j]);
            }
        }

        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            GTEST_FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}
TYPED_TEST(LIBSVMWrite, write_sparse_without_label) {
    using real_type = typename TypeParam::real_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.3 } },
        { real_type{ 2.1 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 3.1 }, real_type{ 3.2 }, real_type{ 0.0 } }
    };

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.size());
    // check the lines
    for (std::size_t i = 0; i < data.size(); ++i) {
        // assemble correct line
        std::string line{};
        for (std::size_t j = 0; j < data[i].size(); ++j) {
            if (data[i][j] != real_type{ 0.0 }) {
                line += fmt::format("{}:{:.10e} ", j + 1, data[i][j]);
            }
        }

        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            GTEST_FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(LIBSVMWrite, empty_data) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{};
    const std::vector<label_type> label{};

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
}

TYPED_TEST(LIBSVMWriteDeathTest, data_with_provided_empty_labels) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{ { real_type{ 1.0 } } };
    const std::vector<label_type> label{};

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_libsvm_data(this->filename, data, label), "has_label is 'true' but no labels were provided!");
}
TYPED_TEST(LIBSVMWriteDeathTest, data_and_label_size_mismatch) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{ { real_type{ 1.0 } }, { real_type{ 2.0 } } };
    const std::vector<label_type> label{ plssvm::detail::convert_to<label_type>("42") };

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_libsvm_data(this->filename, data, label), ::testing::HasSubstr("Number of data points (2) and number of labels (1) mismatch!"));
}
TYPED_TEST(LIBSVMWriteDeathTest, labels_provided_but_not_written) {
    using real_type = typename TypeParam::real_type;
    using label_type = typename TypeParam::label_type;

    // define data to write
    const std::vector<std::vector<real_type>> data{ { real_type{ 1.0 } }, { real_type{ 2.0 } } };
    const std::vector<label_type> label{ plssvm::detail::convert_to<label_type>("42") };

    // try to write the necessary data to the file
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_data_impl<real_type, label_type, false>(this->filename, data, label)), "has_label is 'false' but labels were provided!");
}