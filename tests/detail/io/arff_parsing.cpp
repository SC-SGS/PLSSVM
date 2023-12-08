/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to parsing the data points from a ARFF file.
 */

#include "plssvm/detail/io/arff_parsing.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::label_type_gtest, util::test_parameter_type_at_t
#include "utility.hpp"             // util::{temporary_file, instantiate_template_file, get_correct_data_file_labels, get_distinct_label, generate_specific_matrix}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_P, TYPED_TEST, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, FAIL
                                   // ::testing::{Test, Types, TestWithParam, Values}

#include <cstddef>      // std::size_t
#include <set>          // std::set
#include <string>       // std::string
#include <tuple>        // std::tuple, std::make_tuple, std::ignore
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

class ARFFParseHeader : public ::testing::Test {};
class ARFFParseHeaderValid : public ::testing::TestWithParam<std::tuple<std::string, std::size_t, std::size_t, bool, std::size_t>> {};
TEST_P(ARFFParseHeaderValid, header) {
    const auto &[filename_part, num_features, header_skip, has_label, label_idx] = GetParam();

    // parse the ARFF file
    const std::string filename = fmt::format("{}{}", PLSSVM_TEST_PATH, filename_part);
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    const auto &[parsed_num_features, parsed_header_skip, unique_label, parsed_label_idx] = plssvm::detail::io::parse_arff_header<int>(reader.lines());

    // check for correctness
    EXPECT_EQ(parsed_num_features, num_features);
    EXPECT_EQ(parsed_header_skip, header_skip);
    EXPECT_EQ(!unique_label.empty(), has_label);
    if (has_label) {
        EXPECT_EQ(unique_label, (std::set<int>{ -1, 1 }));
    }
    EXPECT_EQ(parsed_label_idx, label_idx);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ARFFParse, ARFFParseHeaderValid, ::testing::Values(
                                                     std::make_tuple("/data/arff/5x4.arff", 4, 7, true, 4),
                                                     std::make_tuple("/data/arff/5x4_sparse.arff", 4, 7, true, 2),
                                                     std::make_tuple("/data/arff/3x2_without_label.arff", 2, 4, false, 0)));
// clang-format on

TEST(ARFFParseHeader, class_unquoted_nominal_attribute) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_unquoted_nominal_attribute.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(The "@ATTRIBUTE class    0,1" nominal attribute must be enclosed with {}!)");
}
TEST(ARFFParseHeader, class_with_wrong_label) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_with_wrong_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(May not use the combination of the reserved name "class" and attribute type NUMERIC!)");
}
TEST(ARFFParseHeader, class_without_label) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_without_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(The "@ATTRIBUTE class" field must contain class labels!)");
}
TEST(ARFFParseHeader, multiple_classes) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/multiple_classes.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "A nominal attribute with the name CLASS may only be provided once!");
}
TEST(ARFFParseHeader, no_features) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/no_features.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no feature ATTRIBUTES are defined!");
}
TEST(ARFFParseHeader, no_data_attribute) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/no_data_attribute.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: @DATA is missing!");
}
TEST(ARFFParseHeader, nominal_attribute_with_wrong_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/nominal_attribute_with_wrong_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(Read an invalid header entry: "@ATTRIBUTE foo    {0,1}"!)");
}
TEST(ARFFParseHeader, numeric_unquoted) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/numeric_unquoted.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(A "@ATTRIBUTE second entry   numeric" name that contains a whitespace must be quoted!)");
}
TEST(ARFFParseHeader, numeric_without_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/numeric_without_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(The "@ATTRIBUTE   numeric" field must contain a name!)");
}
TEST(ARFFParseHeader, relation_not_at_beginning) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_not_at_beginning.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "The @RELATION attribute must be set before any other @ATTRIBUTE!");
}
TEST(ARFFParseHeader, relation_unquoted) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_unquoted.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(A "@RELATION  name with whitespaces" name that contains a whitespace must be quoted!)");
}
TEST(ARFFParseHeader, relation_without_name) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/relation_without_name.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(The "@RELATION" field must contain a name!)");
}
TEST(ARFFParseHeader, wrong_line) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/wrong_line.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      R"(Read an invalid header entry: "@THIS IS NOT A CORRECT LINE!"!)");
}
TEST(ARFFParseHeader, empty) {
    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_header<int>(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no feature ATTRIBUTES are defined!");
}

template <typename T>
class ARFFParse : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(ARFFParse, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class ARFFParseDense : public ARFFParse<T>, protected util::temporary_file {
  protected:
    using typename ARFFParse<T>::fixture_label_type;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_TEMPLATE.arff", this->filename);
    }

    /**
     * @brief Return the correct dense data points of the template ARFF file.
     * @return the correct data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::soa_matrix<plssvm::real_type> &get_correct_data() const noexcept { return correct_data_; }
    /**
     * @brief Return the correct labels of the template ARFF file.
     * @return the correct labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<fixture_label_type> &get_correct_label() const noexcept { return correct_label_; }

  private:
    /// The correct dense data points.
    plssvm::soa_matrix<plssvm::real_type> correct_data_{ { { plssvm::real_type{ -1.117827500607882 }, plssvm::real_type{ -2.9087188881250993 }, plssvm::real_type{ 0.66638344270039144 }, plssvm::real_type{ 1.0978832703949288 } },
                                                           { plssvm::real_type{ -0.5282118298909262 }, plssvm::real_type{ -0.335880984968183973 }, plssvm::real_type{ 0.51687296029754564 }, plssvm::real_type{ 0.54604461446026 } },
                                                           { plssvm::real_type{ 0.57650218263054642 }, plssvm::real_type{ 1.01405596624706053 }, plssvm::real_type{ 0.13009428079760464 }, plssvm::real_type{ 0.7261913886869387 } },
                                                           { plssvm::real_type{ -0.20981208921241892 }, plssvm::real_type{ 0.60276937379453293 }, plssvm::real_type{ -0.13086851759108944 }, plssvm::real_type{ 0.10805254527169827 } },
                                                           { plssvm::real_type{ 1.88494043717792 }, plssvm::real_type{ 1.00518564317278263 }, plssvm::real_type{ 0.298499933047586044 }, plssvm::real_type{ 1.6464627048813514 } },
                                                           { plssvm::real_type{ -1.1256816275635 }, plssvm::real_type{ 2.12541534341344414 }, plssvm::real_type{ -0.165126576545454511 }, plssvm::real_type{ 2.5164553141200987 } } },
                                                         plssvm::PADDING_SIZE,
                                                         plssvm::PADDING_SIZE };
    /// The correct labels.
    std::vector<fixture_label_type> correct_label_{ util::get_correct_data_file_labels<fixture_label_type>() };
};
TYPED_TEST_SUITE(ARFFParseDense, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class ARFFParseSparse : public ARFFParse<T>, protected util::temporary_file {
  protected:
    using typename ARFFParse<T>::fixture_label_type;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/arff/6x4_sparse_TEMPLATE.arff", this->filename);
    }

    /**
     * @brief Return the correct sparse data points of the template ARFF file.
     * @return the correct data points (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::soa_matrix<plssvm::real_type> &get_correct_data() const noexcept { return correct_data; }
    /**
     * @brief Return the correct labels of the template ARFF file.
     * @return the correct labels (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<fixture_label_type> &get_correct_label() const noexcept { return correct_label; }

  private:
    /// The correct sparse data points.
    plssvm::soa_matrix<plssvm::real_type> correct_data{ { { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.51687296029754564 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                          { plssvm::real_type{ 1.01405596624706053 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                          { plssvm::real_type{ 0.60276937379453293 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.13086851759108944 }, plssvm::real_type{ 0.0 } },
                                                          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.298499933047586044 } },
                                                          { plssvm::real_type{ 0.0 }, plssvm::real_type{ -1.615267454510097261 }, plssvm::real_type{ 2.098278675127757651 }, plssvm::real_type{ 0.0 } } },
                                                        plssvm::PADDING_SIZE,
                                                        plssvm::PADDING_SIZE };
    /// The correct labels.
    std::vector<fixture_label_type> correct_label{ util::get_correct_data_file_labels<fixture_label_type>() };
};
TYPED_TEST_SUITE(ARFFParseSparse, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ARFFParseDense, read) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_arff_data<label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 6);
    ASSERT_EQ(num_features, 4);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());
    EXPECT_EQ(label, this->get_correct_label());
}
TYPED_TEST(ARFFParseSparse, read) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_arff_data<label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 6);
    ASSERT_EQ(num_features, 4);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());
    EXPECT_EQ(label, this->get_correct_label());
}

TYPED_TEST(ARFFParse, read_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/3x2_without_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_arff_data<label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 3);
    ASSERT_EQ(num_features, 2);

    // check for correct data
    const plssvm::soa_matrix<plssvm::real_type> correct_data{ { { plssvm::real_type{ 1.5 }, plssvm::real_type{ -2.9 } },
                                                                { plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.3 } },
                                                                { plssvm::real_type{ 5.5 }, plssvm::real_type{ 0.0 } } },
                                                              plssvm::PADDING_SIZE,
                                                              plssvm::PADDING_SIZE };
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, correct_data);
    EXPECT_TRUE(label.empty());
}

TYPED_TEST(ARFFParse, at_inside_data_section) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/@_inside_data_section.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      R"(Read @ inside data section!: "@ATTRIBUTE invalid numeric"!)");
}
TYPED_TEST(ARFFParse, sparse_missing_closing_brace) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/sparse_missing_closing_brace.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      R"(Missing closing '}' for sparse data point "{2 0.51687296029754564,3 0.54604461446026,4 1" description!)");
}
TYPED_TEST(ARFFParse, sparse_missing_opening_brace) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/sparse_missing_opening_brace.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      R"(Missing opening '{' for sparse data point "1 0.60276937379453293,2 -0.13086851759108944,4 0}" description!)");
}
TYPED_TEST(ARFFParse, sparse_invalid_feature_index) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/sparse_invalid_feature_index.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Trying to add feature/label at index 5 but the maximum index is 4!");
}
TYPED_TEST(ARFFParse, sparse_missing_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/sparse_missing_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      R"(Missing label for data point "{0 1.88494043717792,1 1.00518564317278263,2 0.298499933047586044,3 1.6464627048813514}"!)");
}

TYPED_TEST(ARFFParse, dense_missing_value) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/dense_missing_value.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Invalid number of features and labels! Found 3 but should be 5!");
}
TYPED_TEST(ARFFParse, dense_too_many_values) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/dense_too_many_values.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Invalid number of features and labels! Found 6 but should be 5!");
}
TYPED_TEST(ARFFParse, class_same_label_multiple_times) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_same_label_multiple_times.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Provided 2 labels but only 1 of them was/where unique!");
}
TYPED_TEST(ARFFParse, class_with_only_one_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/class_with_only_one_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Only a single label has been provided!");
}
TYPED_TEST(ARFFParse, usage_of_undefined_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/usage_of_undefined_label.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    if constexpr (!std::is_same_v<label_type, bool>) {
        // it is not possible to have two boolean label and use a third undefined one
        EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                          plssvm::invalid_file_format_exception,
                          R"#(Found the label "2" which was not specified in the header ({0, 1})!)#");
    } else {
        SUCCEED() << "By definition boolean labels do only support two different labels and, therefore, no third undefined one can be used.";
    }
}
TYPED_TEST(ARFFParse, string_label_with_whitespace) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/invalid/string_label_with_whitespace.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('%');
    if constexpr (std::is_same_v<label_type, std::string>) {
        // it is not possible to have two boolean label and use a third undefined one
        EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                          plssvm::invalid_file_format_exception,
                          R"(String labels may not contain whitespaces, but "Hello World" has at least one!)");
    } else {
        SUCCEED() << "By definition non-string labels cannot contain a whitespace.";
    }
}
TYPED_TEST(ARFFParse, libsvm_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)), plssvm::invalid_file_format_exception);
}

template <typename T>
class ARFFParseDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(ARFFParseDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ARFFParseDeathTest, invalid_file_reader) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // open file_reader without associating it to a file
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_arff_data<label_type>(reader)),
                 "The file_reader is currently not associated with a file!");
}

template <typename T>
class ARFFWrite : public ::testing::Test, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(ARFFWrite, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class ARFFWriteDeathTest : public ARFFWrite<T> {};
TYPED_TEST_SUITE(ARFFWriteDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(ARFFWrite, write_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const std::vector<label_type> label = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(label.size(), 3);

    // write the necessary data to the file
    plssvm::detail::io::write_arff_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), 6 + data.num_rows());  // 6 header lines: @RELATION + 3 features + class + @DATA
    // check the header information
    EXPECT_TRUE(plssvm::detail::starts_with(reader.line(0), "@RELATION"));
    for (std::size_t i = 0; i < data.num_cols(); ++i) {
        EXPECT_EQ(reader.line(i + 1), fmt::format("@ATTRIBUTE feature_{} NUMERIC", i));
    }
    EXPECT_EQ(reader.line(4), fmt::format("@ATTRIBUTE class {{{}}}", fmt::join(util::get_distinct_label<label_type>(), ",")));
    EXPECT_EQ(reader.line(5), "@DATA");
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        const std::string line = fmt::format("{:.10e},{:.10e},{:.10e},{}", data(i, 0), data(i, 1), data(i, 2), label[i]);
        bool line_found = false;
        for (std::size_t j = 6; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(ARFFWrite, write_without_label) {
    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 3);

    // write the necessary data to the file
    plssvm::detail::io::write_arff_data(this->filename, data);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), 5 + data.num_rows());  // 6 header lines: @RELATION + 3 features + @DATA
    // check the header information
    EXPECT_TRUE(plssvm::detail::starts_with(reader.line(0), "@RELATION"));
    for (std::size_t i = 0; i < data.num_cols(); ++i) {
        EXPECT_EQ(reader.line(i + 1), fmt::format("@ATTRIBUTE feature_{} NUMERIC", i));
    }
    EXPECT_EQ(reader.line(4), "@DATA");
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        const std::string line = fmt::format("{:.10e},{:.10e},{:.10e}", data(i, 0), data(i, 1), data(i, 2));
        bool line_found = false;
        for (std::size_t j = 5; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(ARFFWrite, empty_data) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{};
    const std::vector<label_type> label{};

    // write the necessary data to the file
    plssvm::detail::io::write_arff_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('%');

    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
}

TYPED_TEST(ARFFWriteDeathTest, data_with_provided_empty_labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 1, 1, plssvm::real_type{ 1.0 } };
    const std::vector<label_type> label{};

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_arff_data(this->filename, data, label), "has_label is 'true' but no labels were provided!");
}
TYPED_TEST(ARFFWriteDeathTest, data_and_label_size_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 2, 1, plssvm::real_type{ 1.0 } };
    const std::vector<label_type> label{ util::get_distinct_label<label_type>().front() };

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_arff_data(this->filename, data, label),
                 ::testing::HasSubstr("Number of data points (2) and number of labels (1) mismatch!"));
}
TYPED_TEST(ARFFWriteDeathTest, labels_provided_but_not_written) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 2, 1, plssvm::real_type{ 1.0 } };
    const std::vector<label_type> label{ util::get_distinct_label<label_type>().front() };

    // try to write the necessary data to the file
    EXPECT_DEATH((plssvm::detail::io::write_arff_data_impl<label_type, false>(this->filename, data, label)),
                 "has_label is 'false' but labels were provided!");
}