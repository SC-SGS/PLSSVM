/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to saving data_set.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::as_lower_case
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::data_set_exception
#include "plssvm/file_format_types.hpp"      // plssvm::file_format_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"             // util::{redirect_output, temporary_file, get_correct_data_file_labels, generate_specific_matrix}

#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{ContainsRegex, StartsWith}
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_TRUE, ASSERT_EQ, ::testing::Test

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::rename
#include <regex>        // std::regex, std::regex::extended, std::regex_match
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

template <typename T>
class DataSetSave : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    /**
     * @brief Return the correct labels used to save a data file.
     * @return the correct label (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<fixture_label_type> &get_label() const noexcept { return label_; }
    /**
     * @brief Return the correct data points used to save a data file.
     * @return the data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::aos_matrix<plssvm::real_type> &get_data_points() const noexcept { return data_points_; }

  private:
    /// The correct labels.
    std::vector<fixture_label_type> label_{ util::get_correct_data_file_labels<fixture_label_type>() };
    /// The correct data points.
    plssvm::aos_matrix<plssvm::real_type> data_points_{ util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label_.size(), 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE) };
};
TYPED_TEST_SUITE(DataSetSave, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetSave, save_invalid_automatic_format) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points(), this->get_label() };

    // try to save to temporary file with an unrecognized extension
    EXPECT_THROW_WHAT(data.save("test.txt"),
                      plssvm::data_set_exception,
                      "Unrecognized file extension for file \"test.txt\" (must be one of: .libsvm or .arff)!");
}

TYPED_TEST(DataSetSave, save_libsvm_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points(), this->get_label() };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::libsvm);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->get_data_points().num_rows());
    const std::regex reg{ ".+ ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}
TYPED_TEST(DataSetSave, save_libsvm_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set without labels
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::libsvm);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->get_data_points().num_rows());
    const std::regex reg{ "([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}
TYPED_TEST(DataSetSave, save_libsvm_automatic_format) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points(), this->get_label() };
    // rename temporary such that it ends with .libsvm
    const std::string old_filename = this->filename;
    this->filename += ".libsvm";
    std::filesystem::rename(old_filename, this->filename);
    // save to temporary file
    data.save(this->filename);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // create regex to check for the correct output
    ASSERT_EQ(reader.num_lines(), this->get_data_points().num_rows());
    const std::regex reg{ ".+ ([0-9]*:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)? ?){4}", std::regex::extended };
    for (const std::string_view line : reader.lines()) {
        EXPECT_TRUE(std::regex_match(std::string{ line }, reg));
    }
}

TYPED_TEST(DataSetSave, save_arff_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points(), this->get_label() };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::arff);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->get_data_points().num_cols();
    const std::size_t expected_header_size = num_features + 3;  // num_features + @RELATION + class + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->get_data_points().num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::ContainsRegex("@attribute class \\{.*,.*\\}"));
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features + 1)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){4}.+", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetSave, save_arff_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points() };
    // save to temporary file
    data.save(this->filename, plssvm::file_format_type::arff);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->get_data_points().num_cols();
    const std::size_t expected_header_size = num_features + 2;  // num_features + @RELATION + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->get_data_points().num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){3}[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}
TYPED_TEST(DataSetSave, save_arff_automatic_format) {
    using label_type = typename TestFixture::fixture_label_type;

    // create data set with labels
    const plssvm::data_set<label_type> data{ this->get_data_points(), this->get_label() };
    // rename temporary such that it ends with .arff
    const std::string old_filename = this->filename;
    this->filename += ".arff";
    std::filesystem::rename(old_filename, this->filename);
    // save to temporary file
    data.save(this->filename);

    // read the file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // create regex to check for the correct output
    const std::size_t num_features = this->get_data_points().num_cols();
    const std::size_t expected_header_size = num_features + 3;  // num_features + @RELATION + class + @DATA
    ASSERT_EQ(reader.num_lines(), expected_header_size + this->get_data_points().num_rows());
    // check header
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(0)), ::testing::StartsWith("@relation"));
    for (std::size_t i = 0; i < num_features; ++i) {
        EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(i + 1)), ::testing::ContainsRegex("@attribute .* numeric"));
    }
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features)), ::testing::ContainsRegex("@attribute class \\{.*,.*\\}"));
    EXPECT_THAT(plssvm::detail::as_lower_case(reader.line(1 + num_features + 1)), ::testing::StartsWith("@data"));
    // check data points
    const std::regex reg{ "([-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?,){4}.+", std::regex::extended };
    for (std::size_t i = expected_header_size; i < reader.num_lines(); ++i) {
        EXPECT_TRUE(std::regex_match(std::string{ reader.line(i) }, reg));
    }
}