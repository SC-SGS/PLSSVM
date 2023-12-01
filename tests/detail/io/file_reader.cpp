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

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::starts_with
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::file_not_found_exception, plssvm::file_reader_exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::{test_parameter_to_name, pretty_print_escaped_string}
#include "types_to_test.hpp"       //  util::{wrap_tuple_types_in_type_lists_t, cartesian_type_product_t, test_parameter_type_at_t};

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, TEST_P, TYPED_TEST, EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_FALSE, ASSERT_TRUE, ASSERT_FALSE, TYPED_TEST_SUITE, INSTANTIATE_TEST_SUITE_P
                          // ::testing::{Test, Types, TestWithParam, ValuesIn}

#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::path
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move, std::swap
#include <vector>       // std::vector

TEST(FileReader, default_construct) {
    // default construct file_reader
    const plssvm::detail::io::file_reader reader{};

    // file reader must not be associated to a file, i.e., is_open must return false
    EXPECT_FALSE(reader.is_open());
    // check other fields for default values
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_EQ(reader.buffer(), nullptr);
}

TEST(FileReader, move_construct) {
    // construct first file_reader
    plssvm::detail::io::file_reader reader1{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    reader1.read_lines('#');

    // move-construct second file_reader
    const plssvm::detail::io::file_reader reader2{ std::move(reader1) };

    // test content of the second file_reader
    EXPECT_TRUE(reader2.is_open());
    EXPECT_EQ(reader2.num_lines(), 5);
    EXPECT_EQ(reader2.lines().size(), 5);
    EXPECT_NE(reader2.buffer(), nullptr);

    // check whether the first file_reader is in a valid moved-from state
    EXPECT_FALSE(reader1.is_open());
    EXPECT_EQ(reader1.num_lines(), 0);
    EXPECT_TRUE(reader1.lines().empty());
    EXPECT_EQ(reader1.buffer(), nullptr);
}

TEST(FileReader, move_assign) {
    // construct first file_reader
    plssvm::detail::io::file_reader reader1{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    reader1.read_lines('#');

    // default-construct second file_reader and move-assign reader1 to it
    plssvm::detail::io::file_reader reader2{};
    reader2 = std::move(reader1);

    // test content of the second file_reader
    EXPECT_TRUE(reader2.is_open());
    EXPECT_EQ(reader2.num_lines(), 5);
    EXPECT_EQ(reader2.lines().size(), 5);
    EXPECT_NE(reader2.buffer(), nullptr);

    // check whether the first file_reader is in a valid moved-from state
    EXPECT_FALSE(reader1.is_open());
    EXPECT_EQ(reader1.num_lines(), 0);
    EXPECT_TRUE(reader1.lines().empty());
    EXPECT_EQ(reader1.buffer(), nullptr);
}

// the input filename types to test
using open_parameter_types = std::tuple<const char *, std::string, std::filesystem::path>;
using open_parameter_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<open_parameter_types>>;

template <typename T>
class FileReaderConstructWithOpen : public ::testing::Test {
  protected:
    using fixture_open_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(FileReaderConstructWithOpen, open_parameter_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(FileReaderConstructWithOpen, non_empty_file) {
    using open_type = typename TestFixture::fixture_open_type;

    // create file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    // construct a file_reader
    const plssvm::detail::io::file_reader reader{ filename };

    EXPECT_TRUE(reader.is_open());  // since we associated a file with reader, a file must be open!
    // since we did not call .read_lines() no PARSED content is available!
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_NE(reader.buffer(), nullptr);
}
TYPED_TEST(FileReaderConstructWithOpen, empty_file) {
    using open_type = typename TestFixture::fixture_open_type;

    // create file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/empty.txt" };
    // construct a file_reader using a c-string literal
    const plssvm::detail::io::file_reader reader{ filename };

    EXPECT_TRUE(reader.is_open());  // since we associated a file with reader, a file must be open!
    // since we did not call .read_lines() no PARSED content is available!
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_EQ(reader.buffer(), nullptr);
}
TYPED_TEST(FileReaderConstructWithOpen, file_not_found) {
    using open_type = typename TestFixture::fixture_open_type;

    // create file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/file_not_found" };
    EXPECT_THROW_WHAT(plssvm::detail::io::file_reader{ filename },
                      plssvm::file_not_found_exception,
                      "Couldn't find file: '" PLSSVM_TEST_PATH "/data/file_not_found'!");
}

template <typename T>
class FileReaderOpen : public ::testing::Test {
  protected:
    using fixture_open_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(FileReaderOpen, open_parameter_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(FileReaderOpen, non_empty_file) {
    using open_type = typename TestFixture::fixture_open_type;

    // create default constructed file reader and open it using the file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    // construct a default file_reader and open a file
    plssvm::detail::io::file_reader reader{};
    reader.open(filename);

    EXPECT_TRUE(reader.is_open());  // since we associated a file with reader, a file must be open!
    // since we did not call .read_lines() no PARSED content is available!
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_NE(reader.buffer(), nullptr);
}
TYPED_TEST(FileReaderOpen, empty_file) {
    using open_type = typename TestFixture::fixture_open_type;

    // create default constructed file reader and open it using the file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/empty.txt" };
    // construct a default file_reader and open a file
    plssvm::detail::io::file_reader reader{};
    reader.open(filename);

    EXPECT_TRUE(reader.is_open());  // since we associated a file with reader, a file must be open!
    // since we did not call .read_lines() no PARSED content is available!
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_EQ(reader.buffer(), nullptr);
}
TYPED_TEST(FileReaderOpen, file_not_found) {
    using open_type = typename TestFixture::fixture_open_type;

    // create default constructed file reader and open it using the file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/file_not_found" };
    // construct a default file_reader and open a file
    plssvm::detail::io::file_reader reader{};
    EXPECT_THROW_WHAT(reader.open(filename),
                      plssvm::file_not_found_exception,
                      "Couldn't find file: '" PLSSVM_TEST_PATH "/data/file_not_found'!");
}
TYPED_TEST(FileReaderOpen, multiple_open) {
    using open_type = typename TestFixture::fixture_open_type;

    // create default constructed file reader and open it using the file name depending on the current test type
    const open_type filename{ PLSSVM_TEST_PATH "/data/empty.txt" };
    // construct a default file_reader and open a file
    plssvm::detail::io::file_reader reader{};
    reader.open(filename);
    // a file must be open
    ASSERT_TRUE(reader.is_open());
    // trying to open another file must throw
    EXPECT_THROW_WHAT(reader.open(filename),
                      plssvm::file_reader_exception,
                      "This file_reader is already associated to a file!");
}

TEST(FileReader, close) {
    // create a new file_reader and associate it to a file
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    ASSERT_TRUE(reader.is_open());

    // close the file
    reader.close();

    // check if the values have been correctly reset
    EXPECT_FALSE(reader.is_open());
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_EQ(reader.buffer(), nullptr);
}
TEST(FileReader, close_twice) {
    // create a new file_reader and associate it to a file
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    ASSERT_TRUE(reader.is_open());

    // close the file twice should do no harm
    reader.close();
    reader.close();

    // check if the values have been correctly reset
    EXPECT_FALSE(reader.is_open());
    EXPECT_EQ(reader.num_lines(), 0);
    EXPECT_TRUE(reader.lines().empty());
    EXPECT_EQ(reader.buffer(), nullptr);
}

TEST(FileReader, is_open) {
    // create a new default constructed file_reader
    plssvm::detail::io::file_reader reader{};
    // noe file must be open
    EXPECT_FALSE(reader.is_open());
    // open a file
    reader.open(PLSSVM_TEST_PATH "/data/empty.txt");
    EXPECT_TRUE(reader.is_open());
    // close the file
    reader.close();
    EXPECT_FALSE(reader.is_open());
}

TEST(FileReader, swap_member_function) {
    // create two file readers
    plssvm::detail::io::file_reader reader1{};
    plssvm::detail::io::file_reader reader2{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    reader2.read_lines('#');

    // swap the two file readers
    reader1.swap(reader2);

    // check both readers for correct values
    EXPECT_TRUE(reader1.is_open());
    EXPECT_EQ(reader1.num_lines(), 5);
    EXPECT_FALSE(reader1.lines().empty());
    EXPECT_NE(reader1.buffer(), nullptr);

    EXPECT_FALSE(reader2.is_open());
    EXPECT_EQ(reader2.num_lines(), 0);
    EXPECT_TRUE(reader2.lines().empty());
    EXPECT_EQ(reader2.buffer(), nullptr);
}

// clang-format off
const auto & get_file_lines() {
    static const std::array<std::tuple<std::basic_string<char>, char, std::vector<std::basic_string_view<char>>>, 3> lines{
    std::make_tuple(PLSSVM_TEST_PATH "/data/arff/5x4.arff", '%', std::vector<std::string_view>{
                                 "% Title",
                                 "% comments",
                                 "@RELATION name",
                                 "@ATTRIBUTE first    NUMERIC",
                                 "@ATTRIBUTE second   numeric",
                                 "@ATTRIBUTE third    Numeric",
                                 "@ATTRIBUTE fourth   NUMERIC",
                                 "@ATTRIBUTE class    {-1,1}",
                                 "@DATA",
                                 "-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,-1",
                                 "-0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026,-1",
                                 "0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387,1",
                                 "-0.20981208921241892,0.60276937379453293,-0.13086851759108944,0.10805254527169827,1",
                                 "1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514,1" }),
    std::make_tuple(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", '#', std::vector<std::string_view>{
                                 "# comment",
                                 "1 1:-1.117827500607882 2:-2.9087188881250993 3:0.66638344270039144 4:1.0978832703949288",
                                 "1 1:-0.5282118298909262 2:-0.335880984968183973 3:0.51687296029754564 4:0.54604461446026",
                                 "-1 1:0.57650218263054642 2:1.01405596624706053 3:0.13009428079760464 4:0.7261913886869387",
                                 "-1 1:-0.20981208921241892 2:0.60276937379453293 3:-0.13086851759108944 4:0.10805254527169827",
                                 "-1 1:1.88494043717792 2:1.00518564317278263 3:0.298499933047586044 4:1.6464627048813514" }),
    std::make_tuple(PLSSVM_TEST_PATH "/data/empty.txt", ' ', std::vector<std::string_view>{}) };
    return lines;
}
// clang-format on

/**
 * @brief Filter @p lines that start with @p filter.
 */
template <typename T>
std::vector<std::string_view> filter_lines(const std::vector<std::string_view> &lines, const T filter) {
    std::vector<std::string_view> filtered_lines;
    for (const std::string_view line : lines) {
        if (!plssvm::detail::starts_with(line, filter)) {
            filtered_lines.push_back(line);
        }
    }
    return filtered_lines;
}

class FileReaderLines : public ::testing::TestWithParam<std::tuple<std::string, char, std::vector<std::string_view>>> {};
TEST_P(FileReaderLines, parse_lines_with_comments) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines();

    // no comments have been filtered (but newlines)
    EXPECT_EQ(reader.lines(), lines);
}
TEST_P(FileReaderLines, parse_lines_without_char_comments) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // comments have been filtered
    EXPECT_EQ(reader.lines(), filter_lines(lines, comment));
}
TEST_P(FileReaderLines, parse_lines_without_string_comments) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines("@ATTRIBUTE");

    // the arff header attributes have been filtered
    EXPECT_EQ(reader.lines(), filter_lines(lines, "@ATTRIBUTE"));
}
TEST_P(FileReaderLines, num_lines) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // check if the read number of lines is correct
    EXPECT_EQ(reader.num_lines(), filter_lines(lines, comment).size());
}
TEST_P(FileReaderLines, line) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // check if the i-th line is correct
    const std::vector<std::string_view> filtered_lines = filter_lines(lines, comment);
    ASSERT_EQ(reader.lines().size(), filtered_lines.size());
    for (std::size_t i = 0; i < filtered_lines.size(); ++i) {
        EXPECT_EQ(reader.line(i), filtered_lines[i]);
    }
}
TEST_P(FileReaderLines, lines) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // check if the read lines are correct
    EXPECT_EQ(reader.lines(), filter_lines(lines, comment));
}
TEST_P(FileReaderLines, buffer_valid) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // check if the read lines are correct
    if (filter_lines(lines, comment).empty()) {
        EXPECT_EQ(reader.buffer(), nullptr);
    } else {
        EXPECT_NE(reader.buffer(), nullptr);
    }
}
INSTANTIATE_TEST_SUITE_P(FileReader, FileReaderLines, ::testing::ValuesIn(get_file_lines()), naming::pretty_print_escaped_string<FileReaderLines>);

TEST(FileReaderLines, parse_lines_without_associated_file) {
    // create file_reader without associating it to a file
    plssvm::detail::io::file_reader reader{};

    // reading lines while no file is associated is impossible!
    EXPECT_THROW_WHAT(reader.read_lines(),
                      plssvm::file_reader_exception,
                      "This file_reader is currently not associated to a file!");
}

class FileReaderLinesDeathTest : public ::testing::TestWithParam<std::tuple<std::string, char, std::vector<std::string_view>>> {};
TEST_P(FileReaderLinesDeathTest, line_out_of_bounce) {
    const auto &[filename, comment, lines] = GetParam();
    // create and read file
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines(comment);

    // check if the i-th line is correct
    ASSERT_EQ(reader.lines().size(), filter_lines(lines, comment).size());
    EXPECT_DEATH(std::ignore = reader.line(reader.num_lines()), fmt::format("Out-of-bounce access!: {} >= {}", reader.num_lines(), reader.num_lines()));
}
INSTANTIATE_TEST_SUITE_P(FileReader, FileReaderLinesDeathTest, ::testing::ValuesIn(get_file_lines()), naming::pretty_print_escaped_string<FileReaderLinesDeathTest>);

TEST(FileReader, swap_free_function) {
    // create two file readers
    plssvm::detail::io::file_reader reader1{};
    plssvm::detail::io::file_reader reader2{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    reader2.read_lines('#');

    // swap the two file readers
    using std::swap;
    swap(reader1, reader2);

    // check both readers for correct values
    EXPECT_TRUE(reader1.is_open());
    EXPECT_EQ(reader1.num_lines(), 5);
    EXPECT_FALSE(reader1.lines().empty());
    EXPECT_NE(reader1.buffer(), nullptr);

    EXPECT_FALSE(reader2.is_open());
    EXPECT_EQ(reader2.num_lines(), 0);
    EXPECT_TRUE(reader2.lines().empty());
    EXPECT_EQ(reader2.buffer(), nullptr);
}