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

#include "plssvm/detail/io/file_reader.hpp"

#include "../../utility.hpp"

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <sstream>  // std::ostringstream
#include <utility>  // std::pair
#include <vector>   // std::vector

template <typename real_type>
struct factors {
    std::size_t feature{};
    real_type lower{};
    real_type upper{};
};

template <typename T>
class ScalingFactorsBase : public ::testing::Test {
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
class ScalingFactorsParsing : public ScalingFactorsBase<T> {};
template <typename T>
class ScalingFactorsParsingDeathTest : public ScalingFactorsBase<T> {};

// the floating point types to test
using open_paramete_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(ScalingFactorsParsing, open_paramete_types);
TYPED_TEST_SUITE(ScalingFactorsParsingDeathTest, open_paramete_types);

TYPED_TEST(ScalingFactorsParsing, write) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ -2.0, 2.0 };
    std::vector<factors<real_type>> scaling_factors{ factors<real_type>{ 0, 1.2, 1.2 }, factors<real_type>{ 1, 0.5, -1.4 }, factors<real_type>{ 2, -1.2, 4.4 } };

    // create temporary file containing the scaling factors
    fmt::ostream out = fmt::output_file(this->filename);
    plssvm::detail::io::write_scaling_factors(out, interval, scaling_factors);
    out.close();

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

TYPED_TEST(ScalingFactorsParsingDeathTest, write_illegal_interval) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ 1, -1 };  // illegal interval!
    std::vector<factors<real_type>> scaling_factors(1);

    // create temporary file containing the scaling factors
    fmt::ostream out = fmt::output_file(this->filename);
    EXPECT_DEATH(plssvm::detail::io::write_scaling_factors(out, interval, scaling_factors), ::testing::HasSubstr("Illegal interval specification: lower (1) < upper (-1)"));
    out.close();
}
TYPED_TEST(ScalingFactorsParsingDeathTest, write_empty_scaling_factors) {
    using real_type = TypeParam;
    // define data to write
    const std::pair<real_type, real_type> interval{ -1.5, 1.5 };
    std::vector<factors<real_type>> scaling_factors{};  // at least one scaling factor must be given!

    // create temporary file containing the scaling factors
    fmt::ostream out = fmt::output_file(this->filename);
    EXPECT_DEATH(plssvm::detail::io::write_scaling_factors(out, interval, scaling_factors), "No scaling factors provided!");
    out.close();
}