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

#include "plssvm/data_set.hpp"               // plssvm::data_set::scaling::factors
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader

#include "../../utility.hpp"  // util::gtest_assert_floating_point_near, EXPECT_THROW_WHAT

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ
                                   // ::testing::{Test, Types}

#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <string>      // std::string
#include <utility>     // std::pair
#include <vector>      // std::vector

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

template <typename T>
class LIBSVMReadDense : public ::testing::Test {
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
TYPED_TEST_SUITE(LIBSVMReadDense, type_combinations_types);

template <typename T>
class LIBSVMReadSparse : public ::testing::Test {
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
TYPED_TEST_SUITE(LIBSVMReadSparse, type_combinations_types);

template <typename T>
class LIBSVMRead : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMRead, type_combinations_types);

template <typename T>
class LIBSVMReadDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(LIBSVMReadDeathTest, type_combinations_types);

TYPED_TEST(LIBSVMReadDense, read) {
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

TYPED_TEST(LIBSVMReadSparse, read) {
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

TYPED_TEST(LIBSVMRead, read_without_label) {
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

TYPED_TEST(LIBSVMRead, zero_based_features) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/zero_based_features.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT((plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception,
                      "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}
TYPED_TEST(LIBSVMRead, arff_file) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/5x4.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW((plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception);
}
TYPED_TEST(LIBSVMRead, empty) {
    using current_real_type = typename TypeParam::real_type;
    using current_label_type = typename TypeParam::label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT((plssvm::detail::io::parse_libsvm_data<current_real_type, current_label_type>(reader)), plssvm::invalid_file_format_exception,
                      "Can't parse file: no data points are given!");
}
