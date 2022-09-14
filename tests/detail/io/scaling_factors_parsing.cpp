/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to parsing the scaling factors from a file.
 */

#include "plssvm/detail/io/scaling_factors_parsing.hpp"

#include "plssvm/data_set.hpp"               // plssvm::data_set::scaling::factors
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader

#include "../../utility.hpp"  // util::gtest_assert_floating_point_near, EXPECT_THROW_WHAT

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ
                                   // ::testing::{Test, Types}

#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <string>      // std::string
#include <tuple>       // std::ignore
#include <utility>     // std::pair
#include <vector>      // std::vector

// typedef nested struct
template <typename T>
using factors = typename plssvm::data_set<T>::scaling::factors;

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class ScalingFactorsReadBase : public ::testing::Test {};

template <typename T>
class ScalingFactorsRead : public ScalingFactorsReadBase<T> {};
TYPED_TEST_SUITE(ScalingFactorsRead, floating_point_types);
template <typename T>
class ScalingFactorsReadDeathTest : public ScalingFactorsReadBase<T> {};
TYPED_TEST_SUITE(ScalingFactorsReadDeathTest, floating_point_types);

TYPED_TEST(ScalingFactorsRead, read) {
    using real_type = TypeParam;
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };
    reader.read_lines('#');
    const auto &[scaling_interval, scaling_factors] = plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader);

    // check for correctness
    // scaling interval
    util::gtest_assert_floating_point_near(scaling_interval.first, real_type{ -1.4 });
    util::gtest_assert_floating_point_near(scaling_interval.second, real_type{ 2.6 });
    // scaling factors
    // note that the parsed scaling factors are zero-based!
    const std::vector<factors<real_type>> correct_scaling_factors{
        factors<real_type>{ 0, 0.0, 1.0 }, factors<real_type>{ 1, 1.1, 2.1 }, factors<real_type>{ 3, 3.3, 4.3 }, factors<real_type>{ 4, 4.4, 5.4 }
    };
    ASSERT_EQ(scaling_factors.size(), correct_scaling_factors.size());
    for (std::size_t i = 0; i < correct_scaling_factors.size(); ++i) {
        EXPECT_EQ(scaling_factors[i].feature, correct_scaling_factors[i].feature);
        util::gtest_assert_floating_point_near(scaling_factors[i].lower, correct_scaling_factors[i].lower, "Wrong lower scaling factor!");
        util::gtest_assert_floating_point_near(scaling_factors[i].upper, correct_scaling_factors[i].upper, "Wrong upper scaling factor!");
    }
}
TYPED_TEST(ScalingFactorsRead, read_no_scaling_factors) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/no_scaling_factors.txt" };
    reader.read_lines('#');
    const auto &[scaling_interval, scaling_factors] = plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader);

    // check for correctness
    // scaling interval
    util::gtest_assert_floating_point_near(scaling_interval.first, real_type{ -1.4 });
    util::gtest_assert_floating_point_near(scaling_interval.second, real_type{ 2.6 });
    // scaling factors -> are empty!
    EXPECT_TRUE(scaling_factors.empty());
}
TYPED_TEST(ScalingFactorsRead, too_many_scaling_interval_values) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_many_scaling_interval_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "The interval to which the data points should be scaled must exactly have two values, but 3 were given!");
}
TYPED_TEST(ScalingFactorsRead, too_few_scaling_interval_values) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_scaling_interval_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "The interval to which the data points should be scaled must exactly have two values, but 1 were given!");
}
TYPED_TEST(ScalingFactorsRead, no_header) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/no_header.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "The first line must only contain an 'x', but is \"-1.4 2.6\"!");
}
TYPED_TEST(ScalingFactorsRead, too_few_lines) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_lines.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "At least two lines must be present, but only 1 were given!");
}
TYPED_TEST(ScalingFactorsRead, empty) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/empty.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "At least two lines must be present, but only 0 were given!");
}
TYPED_TEST(ScalingFactorsRead, too_few_scaling_factor_values) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_scaling_factor_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "Each line must contain exactly three values, but 2 were given!");
}
TYPED_TEST(ScalingFactorsRead, too_many_scaling_factor_values) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_many_scaling_factor_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "Each line must contain exactly three values, but 4 were given!");
}
TYPED_TEST(ScalingFactorsRead, zero_based_scaling_factors) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/zero_based_scaling_factors.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), plssvm::invalid_file_format_exception, "The scaling factors must be provided one-based, but are zero-based!");
}
TYPED_TEST(ScalingFactorsRead, invalid_number) {
    using real_type = TypeParam;

    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/invalid_number.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), std::runtime_error, fmt::format("Can't convert 'a' to a value of type {}!", plssvm::detail::arithmetic_type_name<real_type>()));
}

TYPED_TEST(ScalingFactorsReadDeathTest, invalid_file_reader) {
    using real_type = TypeParam;
    // define data to write
    std::pair<real_type, real_type> interval{ -1.5, 1.5 };
    std::vector<factors<real_type>> scaling_factors{ factors<real_type>{} };  // at least one scaling factor must be given!

    // create temporary file containing the scaling factors
    plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_scaling_factors<real_type, factors<real_type>>(reader)), "The file_reader is currently not associated with a file!");
}

template <typename T>
class ScalingFactorsWriteBase : public ::testing::Test {
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
class ScalingFactorsWrite : public ScalingFactorsWriteBase<T> {};
TYPED_TEST_SUITE(ScalingFactorsWrite, floating_point_types);

template <typename T>
class ScalingFactorsWriteDeathTest : public ScalingFactorsWriteBase<T> {};
TYPED_TEST_SUITE(ScalingFactorsWriteDeathTest, floating_point_types);

TYPED_TEST(ScalingFactorsWrite, write) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ -2.0, 2.0 };
    std::vector<factors<real_type>> scaling_factors{ factors<real_type>{ 0, 1.2, 1.2 }, factors<real_type>{ 1, 0.5, -1.4 }, factors<real_type>{ 2, -1.2, 4.4 } };

    // write the necessary data to the file
    plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines();

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), scaling_factors.size() + 2);
    // first line is always the single character x
    EXPECT_EQ(reader.line(0), "x");
    // second line contains the scaling interval
    EXPECT_EQ(reader.line(1), "-2 2");
    // the following lines contain the scaling factors for each feature; note the one-based indexing
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        EXPECT_EQ(reader.line(i + 2), fmt::format("{} {} {}", scaling_factors[i].feature + 1, scaling_factors[i].lower, scaling_factors[i].upper));
    }
}

TYPED_TEST(ScalingFactorsWriteDeathTest, write_illegal_interval) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ 1, -1 };  // illegal interval!
    std::vector<factors<real_type>> scaling_factors(1);

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors), ::testing::HasSubstr("Illegal interval specification: lower (1) < upper (-1)"));
}
TYPED_TEST(ScalingFactorsWriteDeathTest, write_empty_scaling_factors) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ -1.5, 1.5 };
    std::vector<factors<real_type>> scaling_factors{};  // at least one scaling factor must be given!

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors), "No scaling factors provided!");
}