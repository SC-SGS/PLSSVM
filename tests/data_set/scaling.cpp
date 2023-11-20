/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the data_set scaling used for learning an SVM model.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"                 // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::data_set_exception

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"             // util::{temporary_file, redirect_output}

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, ASSERT_EQ, ASSERT_GE , ::testing::Test

#include <cstddef>  // std::size_t
#include <regex>    // std::regex, std::regex::extended, std::regex_match
#include <vector>   // std::vector

template <typename T>
class DataSetScaling : public ::testing::Test, private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(DataSetScaling, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetScaling, default_construct_factor) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    const factor_type factor{};

    // test values
    EXPECT_EQ(factor.feature, std::size_t{});
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{});
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{});
}
TYPED_TEST(DataSetScaling, construct_factor) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factor_type = typename scaling_type::factors;

    // create factor
    const factor_type factor{ 1, plssvm::real_type{ -2.5 }, plssvm::real_type{ 2.5 } };

    // test values
    EXPECT_EQ(factor.feature, 1);
    EXPECT_FLOATING_POINT_EQ(factor.lower, plssvm::real_type{ -2.5 });
    EXPECT_FLOATING_POINT_EQ(factor.upper, plssvm::real_type{ 2.5 });
}

TYPED_TEST(DataSetScaling, construct_interval) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // test whether the values have been correctly set
    EXPECT_FLOATING_POINT_EQ(scale.scaling_interval.first, plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(scale.scaling_interval.second, plssvm::real_type{ 1.0 });
    EXPECT_TRUE(scale.scaling_factors.empty());
}
TYPED_TEST(DataSetScaling, construct_invalid_interval) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class with an invalid interval
    EXPECT_THROW_WHAT((scaling_type{ plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } }),
                      plssvm::data_set_exception,
                      "Inconsistent scaling interval specification: lower (1) must be less than upper (-1)!");
}
TYPED_TEST(DataSetScaling, construct_from_file) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;
    using factors_type = typename scaling_type::factors;

    // create scaling class
    scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // test whether the values have been correctly set
    EXPECT_EQ(scale.scaling_interval.first, plssvm::detail::convert_to<plssvm::real_type>("-1.4"));
    EXPECT_EQ(scale.scaling_interval.second, plssvm::detail::convert_to<plssvm::real_type>("2.6"));
    const std::vector<factors_type> correct_factors = {
        factors_type{ 0, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } },
        factors_type{ 1, plssvm::real_type{ 1.1 }, plssvm::real_type{ 2.1 } },
        factors_type{ 3, plssvm::real_type{ 3.3 }, plssvm::real_type{ 4.3 } },
        factors_type{ 4, plssvm::real_type{ 4.4 }, plssvm::real_type{ 5.4 } },
    };
    ASSERT_EQ(scale.scaling_factors.size(), correct_factors.size());
    for (std::size_t i = 0; i < correct_factors.size(); ++i) {
        EXPECT_EQ(scale.scaling_factors[i].feature, correct_factors[i].feature);
        EXPECT_FLOATING_POINT_EQ(scale.scaling_factors[i].lower, correct_factors[i].lower);
        EXPECT_FLOATING_POINT_EQ(scale.scaling_factors[i].upper, correct_factors[i].upper);
    }
}

TYPED_TEST(DataSetScaling, save) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ PLSSVM_TEST_PATH "/data/scaling_factors/scaling_factors.txt" };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scale.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check file content
    ASSERT_GE(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
    reg = std::regex{ "\\+?[1-9]+[0-9]* [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    for (std::size_t i = 2; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetScaling, save_empty_scaling_factors) {
    using label_type = typename TestFixture::fixture_label_type;
    using scaling_type = typename plssvm::data_set<label_type>::scaling;

    // create scaling class
    const scaling_type scale{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save scaling factors
    scale.save(tmp_file.filename);

    // read file and check its content
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // check the content
    ASSERT_EQ(reader.num_lines(), 2);
    EXPECT_EQ(reader.line(0), "x");
    const std::regex reg{ "[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    EXPECT_TRUE(std::regex_match(std::string{ reader.line(1) }, reg));
}