/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the performance tracker class.
 */

#include "plssvm/detail/performance_tracker.hpp"

#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader

#include "naming.hpp"         // naming::test_parameter_to_name
#include "types_to_test.hpp"  // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"        // util::redirect_output

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // EXPECT_THAT, ::testing::{HasSubstr}
#include "gtest/gtest.h"           // TEST, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test

#include <iostream>  // std::cout, std::clog
#include <string>    // std::string

template <typename T>
class TrackingEntry : public ::testing::Test, public util::redirect_output<> {
  protected:
    using fixture_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(TrackingEntry, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(TrackingEntry, construct) {
    using type = typename TestFixture::fixture_type;

    // construct a tracking entry
    const plssvm::detail::tracking_entry e{ "category", "name", type{} };

    // check the values
    EXPECT_EQ(e.entry_category, "category");
    EXPECT_EQ(e.entry_name, "name");
    EXPECT_EQ(e.entry_value, type{});
}

TYPED_TEST(TrackingEntry, output_operator) {
    using type = typename TestFixture::fixture_type;

    // construct a tracking entry
    const plssvm::detail::tracking_entry e{ "category", "name", type{} };

    // output the value
    std::cout << e;

    // check the output value
    EXPECT_EQ(this->get_capture(), fmt::format("{}", type{}));
}

TEST(TrackingEntry, is_tracking_entry) {
    // check whether the provided type is a tracking entry or not, ignoring any top-level const, reference, and volatile qualifiers
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<int>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<int>>);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<const int &>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<const int &>>);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry<plssvm::detail::tracking_entry<std::string>>::value);
    EXPECT_TRUE(plssvm::detail::is_tracking_entry_v<plssvm::detail::tracking_entry<std::string>>);
}
TEST(TrackingEntry, is_no_tracking_entry) {
    // the following types are NOT tracking entries
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<int>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<int>);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<const int &>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<const int &>);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry<std::string>::value);
    EXPECT_FALSE(plssvm::detail::is_tracking_entry_v<std::string>);
}

class PerformanceTracker : public ::testing::Test, public util::redirect_output<&std::clog> {
  protected:
    void TearDown() override {
        // clear possible tracking entries stored from previous tests
        plssvm::detail::global_tracker->clear_tracking_entries();
    }
};

// the macros are only available if PLSSVM_PERFORMANCE_TRACKER_ENABLED is defined!
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

TEST_F(PerformanceTracker, pause_and_resume_macros) {
    // tracking is enabled per default
    EXPECT_TRUE(plssvm::detail::global_tracker->is_tracking());
    // disable performance tracking
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE();
    // tracking should now be disabled
    EXPECT_FALSE(plssvm::detail::global_tracker->is_tracking());
    // re-enable performance tracking
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME();
    // tracking should now be enabled again
    EXPECT_TRUE(plssvm::detail::global_tracker->is_tracking());
}
TEST_F(PerformanceTracker, add_entry_macro) {
    // add different tracking entries
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "foo", "bar", 42 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "", "foobar", 'a' }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "", "foobar", 'b' }));

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 2);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "42");
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);
    EXPECT_EQ(entries.at("foo").at("baz").front(), "3.1415");

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
    EXPECT_EQ(entries.at("").at("foobar"), (std::vector<std::string>{ "a", "b" }));
}

TEST_F(PerformanceTracker, save_macro) {
    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // read the file
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines();

    // test file contents
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must be empty now
    EXPECT_TRUE(plssvm::detail::global_tracker->get_tracking_entries().empty());
}

#endif

TEST_F(PerformanceTracker, pause_and_resume) {
    // tracking is enabled per default
    EXPECT_TRUE(plssvm::detail::global_tracker->is_tracking());
    // disable performance tracking
    plssvm::detail::global_tracker->pause_tracking();
    // tracking should now be disabled
    EXPECT_FALSE(plssvm::detail::global_tracker->is_tracking());
    // re-enable performance tracking
    plssvm::detail::global_tracker->resume_tracking();
    // tracking should now be enabled again
    EXPECT_TRUE(plssvm::detail::global_tracker->is_tracking());
}
TEST_F(PerformanceTracker, add_generic_tracking_entry) {
    // add different tracking entries
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 2);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "42");
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);
    EXPECT_EQ(entries.at("foo").at("baz").front(), "3.1415");

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
    EXPECT_EQ(entries.at("").at("foobar"), (std::vector<std::string>{ "a", "b" }));
}
TEST_F(PerformanceTracker, add_string_tracking_entry) {
    // add a tracking entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", std::string{ "baz" } });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("foo").size(), 1);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "\"baz\"");
}
TEST_F(PerformanceTracker, add_vector_tracking_entry) {
    // add a tracking entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", std::vector{ 1, 2, 3 } });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", std::vector{ std::string{ "a" }, std::string{ "b" } } });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("foo").size(), 1);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 2);
    EXPECT_EQ(entries.at("foo").at("bar"), (std::vector<std::string>{ "[1, 2, 3]", "[\"a\", \"b\"]" }));
}
TEST_F(PerformanceTracker, add_parameter_tracking_entry) {
    // add a tracking entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", plssvm::parameter{} });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 6);
}
TEST_F(PerformanceTracker, add_parser_train_tracking_entry) {
    // create a parameter train object
    constexpr int argc = 3;
    char argv_arr[argc][20] = { "./plssvm-train", "/path/to/train", "/path/to/model" };
    char *argv[]{ argv_arr[0], argv_arr[1], argv_arr[2] };
    const plssvm::detail::cmd::parser_train parser{ argc, static_cast<char **>(argv) };

    // save cmd::parser_train entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 17);
}
TEST_F(PerformanceTracker, add_parser_predict_tracking_entry) {
    // create a parameter train object
    constexpr int argc = 4;
    char argv_arr[argc][20] = { "./plssvm-predict", "/path/to/train", "/path/to/model", "/path/to/predict" };
    char *argv[]{ argv_arr[0], argv_arr[1], argv_arr[2], argv_arr[3] };
    const plssvm::detail::cmd::parser_predict parser{ argc, static_cast<char **>(argv) };

    // save cmd::parser_predict entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 9);
}
TEST_F(PerformanceTracker, add_parser_scale_tracking_entry) {
    // create a parameter train object
    constexpr int argc = 3;
    char argv_arr[argc][20] = { "./plssvm-train", "/path/to/train", "/path/to/scaled" };
    char *argv[]{ argv_arr[0], argv_arr[1], argv_arr[2] };
    const plssvm::detail::cmd::parser_scale parser{ argc, static_cast<char **>(argv) };

    // save cmd::parser_scale entry
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 10);
}

TEST_F(PerformanceTracker, save_no_additional_entries) {
    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    plssvm::detail::global_tracker->save(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // the tracking entries must be empty now
    EXPECT_TRUE(plssvm::detail::global_tracker->get_tracking_entries().empty());
}
TEST_F(PerformanceTracker, save_entries_to_file) {
    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });
    plssvm::detail::global_tracker->save(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // read the file
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // test file contents
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must be empty now
    EXPECT_TRUE(plssvm::detail::global_tracker->get_tracking_entries().empty());
}
TEST_F(PerformanceTracker, save_entries_empty_file) {
    // save entries to file
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });
    // save to empty file, i.e., dump the performance tracking entries to std::clog
    plssvm::detail::global_tracker->save("");

    // the captured standard output must not be empty
    EXPECT_FALSE(this->get_capture().empty());

    // test file contents
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must be empty now
    EXPECT_TRUE(plssvm::detail::global_tracker->get_tracking_entries().empty());
}

TEST_F(PerformanceTracker, get_tracking_entries) {
    // add different tracking entries
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::global_tracker->get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 2);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
}

TEST_F(PerformanceTracker, clear_tracking_entries) {
    // add different tracking entries
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::global_tracker->add_tracking_entry(plssvm::detail::tracking_entry{ "", "foobar", 'b' });

    // the performance tracker should contain entries
    EXPECT_FALSE(plssvm::detail::global_tracker->get_tracking_entries().empty());

    // clear all tracking entries
    plssvm::detail::global_tracker->clear_tracking_entries();

    // the performance tracker should now contain no tracking entries
    EXPECT_TRUE(plssvm::detail::global_tracker->get_tracking_entries().empty());
}