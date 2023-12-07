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

#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_NEAR, EXPECT_FLOATING_POINT_VECTOR_NEAR, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"             // util::{temporary_file, instantiate_template_file, get_correct_data_file_labels, get_distinct_label, generate_specific_matrix, generate_specific_sparse_matrix}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TEST_P, TYPED_TEST, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_DEATH, ASSERT_EQ, FAIL
                                   // ::testing::{Test, TestWithParam, Values}

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::ignore
#include <utility>  // std::pair, std::make_pair
#include <vector>   // std::vector

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
                                                      std::make_pair("/data/libsvm/5x4.libsvm", 4),
                                                      std::make_pair("/data/libsvm/5x4_sparse.libsvm", 4),
                                                      std::make_pair("/data/libsvm/3x2_without_label.libsvm", 2),
                                                      std::make_pair("/data/libsvm/500x200.libsvm", 200),
                                                      std::make_pair("/data/empty.txt", 0)));
// clang-format on

TEST(LIBSVMParseNumFeatures, index_with_alpha_char_at_the_beginning) {
    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/index_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_num_features(reader.lines())),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' !2' to a value of type unsigned long!");
}

template <typename T>
class LIBSVMParseDense : public ::testing::Test, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", this->filename);
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
TYPED_TEST_SUITE(LIBSVMParseDense, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class LIBSVMParseSparse : public ::testing::Test, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        // create file used in this test fixture by instantiating the template file
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_sparse_TEMPLATE.libsvm", this->filename);
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
    /// The correct sparse data points.
    plssvm::soa_matrix<plssvm::real_type> correct_data_{ { { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                           { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.51687296029754564 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                           { plssvm::real_type{ 1.01405596624706053 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                           { plssvm::real_type{ 0.60276937379453293 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ -0.13086851759108944 }, plssvm::real_type{ 0.0 } },
                                                           { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.298499933047586044 } },
                                                           { plssvm::real_type{ 0.0 }, plssvm::real_type{ -1.615267454510097261 }, plssvm::real_type{ 2.098278675127757651 }, plssvm::real_type{ 0.0 } } },
                                                         plssvm::PADDING_SIZE,
                                                         plssvm::PADDING_SIZE };
    /// The correct labels.
    std::vector<fixture_label_type> correct_label_{ util::get_correct_data_file_labels<fixture_label_type>() };
};
TYPED_TEST_SUITE(LIBSVMParseSparse, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class LIBSVMParse : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(LIBSVMParse, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMParseDense, read) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 6);
    ASSERT_EQ(num_features, 4);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());
    EXPECT_EQ(label, this->get_correct_label());
}

TYPED_TEST(LIBSVMParseSparse, read) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<label_type>(reader);

    // check for correct sizes
    ASSERT_EQ(num_data_points, 6);
    ASSERT_EQ(num_features, 4);

    // check for correct data
    EXPECT_FLOATING_POINT_MATRIX_NEAR(data, this->get_correct_data());
    EXPECT_EQ(label, this->get_correct_label());
}

TYPED_TEST(LIBSVMParse, read_without_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    const auto [num_data_points, num_features, data, label] = plssvm::detail::io::parse_libsvm_data<label_type>(reader);

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

TYPED_TEST(LIBSVMParse, zero_based_features) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/zero_based_features.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!");
}
TYPED_TEST(LIBSVMParse, arff_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the ARFF file
    const std::string filename = PLSSVM_TEST_PATH "/data/arff/5x4.arff";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)), plssvm::invalid_file_format_exception);
}
TYPED_TEST(LIBSVMParse, empty) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/empty.txt";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Can't parse file: no data points are given!");
}
TYPED_TEST(LIBSVMParse, feature_with_alpha_char_at_the_beginning) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/feature_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert 'a-1.11' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TYPED_TEST(LIBSVMParse, index_with_alpha_char_at_the_beginning) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/index_with_alpha_char_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' !2' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, invalid_colon_at_the_beginning) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/invalid_colon_at_the_beginning.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert '' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, invalid_colon_in_the_middle) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/invalid_colon_in_the_middle.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' :2' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, missing_feature_value) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/missing_feature_value.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}
TYPED_TEST(LIBSVMParse, missing_index_value) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/missing_index_value.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Can't convert ' ' to a value of type unsigned long!");
}
TYPED_TEST(LIBSVMParse, inconsistent_label_specification) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/inconsistent_label_specification.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "Inconsistent label specification found (some data points are labeled, others are not)!");
}
TYPED_TEST(LIBSVMParse, non_increasing_indices) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/non_increasing_indices.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 3 is smaller or equal than 3!");
}
TYPED_TEST(LIBSVMParse, non_strictly_increasing_indices) {
    using label_type = typename TestFixture::fixture_label_type;

    // parse the LIBSVM file
    const std::string filename = PLSSVM_TEST_PATH "/data/libsvm/invalid/non_strictly_increasing_indices.libsvm";
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');
    EXPECT_THROW_WHAT(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                      plssvm::invalid_file_format_exception,
                      "The features indices must be strictly increasing, but 2 is smaller or equal than 3!");
}

template <typename T>
class LIBSVMParseDeathTest : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(LIBSVMParseDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMParseDeathTest, invalid_file_reader) {
    using label_type = typename TestFixture::fixture_label_type;

    // open file_reader without associating it to a file
    const plssvm::detail::io::file_reader reader{};
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::parse_libsvm_data<label_type>(reader)),
                 "The file_reader is currently not associated with a file!");
}

template <typename T>
class LIBSVMWrite : public ::testing::Test, protected util::temporary_file {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(LIBSVMWrite, util::label_type_gtest, naming::test_parameter_to_name);

template <typename T>
class LIBSVMWriteDeathTest : public LIBSVMWrite<T> {};
TYPED_TEST_SUITE(LIBSVMWriteDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMWrite, write_dense_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const std::vector<label_type> label = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(label.size(), 3);

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.num_rows());
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        const std::string line = fmt::format("{} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", label[i], data(i, 0), data(i, 1), data(i, 2));
        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}
TYPED_TEST(LIBSVMWrite, write_dense_without_label) {
    // define data to write
    const auto data = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 3);

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.num_rows());
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        const std::string line = fmt::format("1:{:.10e} 2:{:.10e} 3:{:.10e} ", data(i, 0), data(i, 1), data(i, 2));
        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(LIBSVMWrite, write_sparse_with_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const std::vector<label_type> label = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_specific_sparse_matrix<plssvm::soa_matrix<plssvm::real_type>>(label.size(), 3);

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.num_rows());
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        // assemble correct line
        std::string line = fmt::format("{} ", label[i]);
        for (std::size_t j = 0; j < data.num_cols(); ++j) {
            if (data(i, j) != plssvm::real_type{ 0.0 }) {
                line += fmt::format("{}:{:.10e} ", j + 1, data(i, j));
            }
        }

        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}
TYPED_TEST(LIBSVMWrite, write_sparse_without_label) {
    // define data to write
    const auto data = util::generate_specific_sparse_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 3);

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check if the correct number of lines have been read
    ASSERT_EQ(reader.num_lines(), data.num_rows());
    // check the lines
    for (std::size_t i = 0; i < data.num_rows(); ++i) {
        // assemble correct line
        std::string line{};
        for (std::size_t j = 0; j < data.num_cols(); ++j) {
            if (data(i, j) != plssvm::real_type{ 0.0 }) {
                line += fmt::format("{}:{:.10e} ", j + 1, data(i, j));
            }
        }

        bool line_found = false;
        for (std::size_t j = 0; j < reader.num_lines(); ++j) {
            if (reader.line(j) == line) {
                line_found = true;
            }
        }
        if (!line_found) {
            FAIL() << fmt::format("Couldn't find line '{}' in the output file.", line);
        }
    }
}

TYPED_TEST(LIBSVMWrite, empty_data) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{};
    const std::vector<label_type> label{};

    // write the necessary data to the file
    plssvm::detail::io::write_libsvm_data(this->filename, data, label);

    // read the previously written file to check for correctness
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
}

TYPED_TEST(LIBSVMWriteDeathTest, data_with_provided_empty_labels) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 1, 1, 1 };
    const std::vector<label_type> label{};

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_libsvm_data(this->filename, data, label), "has_label is 'true' but no labels were provided!");
}
TYPED_TEST(LIBSVMWriteDeathTest, data_and_label_size_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 2, 1, 1 };
    const std::vector<label_type> label{ util::get_distinct_label<label_type>().front() };

    // try to write the necessary data to the file
    EXPECT_DEATH(plssvm::detail::io::write_libsvm_data(this->filename, data, label),
                 ::testing::HasSubstr("Number of data points (2) and number of labels (1) mismatch!"));
}
TYPED_TEST(LIBSVMWriteDeathTest, labels_provided_but_not_written) {
    using label_type = typename TestFixture::fixture_label_type;

    // define data to write
    const plssvm::soa_matrix<plssvm::real_type> data{ 2, 1, 1 };
    const std::vector<label_type> label{ util::get_distinct_label<label_type>().front() };

    // try to write the necessary data to the file
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_data_impl<label_type, false>(this->filename, data, label)),
                 "has_label is 'false' but labels were provided!");
}