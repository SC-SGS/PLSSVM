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

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/data_set.hpp"               // plssvm::data_set::scaling::factors
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ, EXPECT_THROW_WHAT
#include "utility.hpp"             // util::temporary_file

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_F, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, ::testing::Test

#include <cstddef>    // std::size_t
#include <stdexcept>  // std::runtime_error
#include <tuple>      // std::ignore
#include <utility>    // std::pair
#include <vector>     // std::vector

// typedef nested struct
using factors_type = plssvm::data_set<>::scaling::factors;

TEST(ScalingFactorsRead, read) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };
    reader.read_lines('#');
    const auto &[scaling_interval, scaling_factors] = plssvm::detail::io::parse_scaling_factors<factors_type>(reader);

    // check for correctness
    // scaling interval
    EXPECT_FLOATING_POINT_EQ(scaling_interval.first, plssvm::real_type{ -1.4 });
    EXPECT_FLOATING_POINT_EQ(scaling_interval.second, plssvm::real_type{ 2.6 });
    // scaling factors
    // note that the parsed scaling factors are zero-based!
    const std::vector<factors_type> correct_scaling_factors{
        factors_type{ 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } },
        factors_type{ 1, plssvm::real_type{ 1.1 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.3 }, plssvm::real_type{ 4.3 } },
        factors_type{ 4, plssvm::real_type{ 4.4 }, plssvm::real_type{ 5.4 } }
    };
    ASSERT_EQ(scaling_factors.size(), correct_scaling_factors.size());
    for (std::size_t i = 0; i < correct_scaling_factors.size(); ++i) {
        EXPECT_EQ(scaling_factors[i].feature, correct_scaling_factors[i].feature);
        EXPECT_FLOATING_POINT_EQ(scaling_factors[i].lower, correct_scaling_factors[i].lower);
        EXPECT_FLOATING_POINT_EQ(scaling_factors[i].upper, correct_scaling_factors[i].upper);
    }
}
TEST(ScalingFactorsRead, read_no_scaling_factors) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/no_scaling_factors.txt" };
    reader.read_lines('#');
    const auto &[scaling_interval, scaling_factors] = plssvm::detail::io::parse_scaling_factors<factors_type>(reader);

    // check for correctness
    // scaling interval
    EXPECT_FLOATING_POINT_EQ(scaling_interval.first, plssvm::real_type{ -1.4 });
    EXPECT_FLOATING_POINT_EQ(scaling_interval.second, plssvm::real_type{ 2.6 });
    // scaling factors -> are empty!
    EXPECT_TRUE(scaling_factors.empty());
}
TEST(ScalingFactorsRead, too_many_scaling_interval_values) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_many_scaling_interval_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "The interval to which the data points should be scaled must exactly have two values, but 3 were given!");
}
TEST(ScalingFactorsRead, too_few_scaling_interval_values) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_scaling_interval_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "The interval to which the data points should be scaled must exactly have two values, but 1 were given!");
}
TEST(ScalingFactorsRead, inconsistent_scaling_interval_values) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/inconsistent_scaling_interval_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Inconsistent scaling interval specification: lower (1.4) must be less than upper (-2.6)!");
}
TEST(ScalingFactorsRead, no_header) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/no_header.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      R"(The first line must only contain an 'x', but is "-1.4 2.6"!)");
}
TEST(ScalingFactorsRead, too_few_lines) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_lines.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "At least two lines must be present, but only 1 were given!");
}
TEST(ScalingFactorsRead, empty) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/empty.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "At least two lines must be present, but only 0 were given!");
}
TEST(ScalingFactorsRead, too_few_scaling_factor_values) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_few_scaling_factor_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Each line must contain exactly three values, but 2 were given!");
}
TEST(ScalingFactorsRead, too_many_scaling_factor_values) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/too_many_scaling_factor_values.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Each line must contain exactly three values, but 4 were given!");
}
TEST(ScalingFactorsRead, zero_based_scaling_factors) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/zero_based_scaling_factors.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "The scaling factors must be provided one-based, but are zero-based!");
}
TEST(ScalingFactorsRead, invalid_number) {
    // parse scaling factors!
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/scaling_factors/invalid/invalid_number.txt" };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                      std::runtime_error,
                      fmt::format("Can't convert 'a' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}

TEST(ScalingFactorsReadDeathTest, invalid_file_reader) {
    // create temporary file containing the scaling factors
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_scaling_factors<factors_type>(reader)),
                 "The file_reader is currently not associated with a file!");
}

class ScalingFactorsWrite : public ::testing::Test, protected util::temporary_file {};
class ScalingFactorsWriteDeathTest : public ScalingFactorsWrite {};

TEST_F(ScalingFactorsWrite, write) {
    // define data to write
    const std::pair<plssvm::real_type, plssvm::real_type> interval{ plssvm::real_type{ -2.0 }, plssvm::real_type{ 2.0 } };
    std::vector<factors_type> scaling_factors{
        factors_type{ 0, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.2 } },
        factors_type{ 1, plssvm::real_type{ 0.5 }, plssvm::real_type{ -1.4 } },
        factors_type{ 2, plssvm::real_type{ -1.2 }, plssvm::real_type{ 4.4 } }
    };

    // write the necessary data to the file
    plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), scaling_factors.size() + 2);
    // first line is always the single character x
    EXPECT_EQ(reader.line(0), "x");
    // second line contains the scaling interval
    EXPECT_EQ(reader.line(1), "-2 2");
    // the following lines contain the scaling factors for each feature; note the one-based indexing
    for (std::size_t i = 0; i < scaling_factors.size(); ++i) {
        EXPECT_EQ(reader.line(i + 2), fmt::format("{} {:.10e} {:.10e}", scaling_factors[i].feature + 1, scaling_factors[i].lower, scaling_factors[i].upper));
    }
}
TEST_F(ScalingFactorsWrite, write_empty_scaling_factors) {
    // define data to write
    const std::pair<plssvm::real_type, plssvm::real_type> interval{ plssvm::real_type{ -1.5 }, plssvm::real_type{ 1.5 } };
    const std::vector<factors_type> scaling_factors{};  // write no scaling factors to the file (allowed, but nonsensical)

    // try to write the necessary data to the file
    plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), 2);  // only the header information are saved (x + scaling interval)
    // first line is always the single character x
    EXPECT_EQ(reader.line(0), "x");
    // second line contains the scaling interval
    EXPECT_EQ(reader.line(1), fmt::format("{} {}", -1.5, 1.5));
}

TEST_F(ScalingFactorsWriteDeathTest, write_illegal_interval) {
    // define data to write
    const std::pair<plssvm::real_type, plssvm::real_type> interval{ plssvm::real_type{ 1 }, plssvm::real_type{ -1 } };  // illegal interval!
    const std::vector<factors_type> scaling_factors(1);

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_scaling_factors(this->filename, interval, scaling_factors),
                 ::testing::HasSubstr("Illegal interval specification: lower (1) < upper (-1)"));
}