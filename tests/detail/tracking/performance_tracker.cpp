/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the performance tracker class.
 */

#include "plssvm/detail/tracking/performance_tracker.hpp"

#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train
#include "plssvm/detail/io/file_reader.hpp"      // plssvm::detail::io::file_reader
#include "plssvm/detail/memory_size.hpp"         // plssvm::detail::memory_size (literals)
#include "plssvm/detail/tracking/events.hpp"     // plssvm::detail::tracking::{events, event}

#include "tests/detail/tracking/mock_hardware_sampler.hpp"  // mock_hardware_sampler
#include "tests/naming.hpp"                                 // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"                          // util::{label_type_gtest, test_parameter_type_at_t}
#include "tests/utility.hpp"                                // util::redirect_output

#include "fmt/core.h"     // fmt::format
#include "gmock/gmock.h"  // EXPECT_CALL, EXPECT_THAT, ::testing::{HasSubstr}
#include "gtest/gtest.h"  // TEST, TYPED_TEST_SUITE, TYPED_TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test, ::testing::An

#include <algorithm>   // std::transform
#include <array>       // std::array
#include <chrono>      // std::chrono_literals namespace
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::is_empty
#include <iostream>    // std::cout, std::clog
#include <map>         // std::map
#include <string>      // std::string
#include <vector>      // std::vector

using namespace plssvm::detail::literals;

template <typename T>
class TrackingEntry : public ::testing::Test,
                      public util::redirect_output<> {
  protected:
    using fixture_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(TrackingEntry, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(TrackingEntry, construct) {
    using type = typename TestFixture::fixture_type;

    // construct a tracking entry
    const plssvm::detail::tracking::tracking_entry e{ "category", "name", type{} };

    // check the values
    EXPECT_EQ(e.entry_category, "category");
    EXPECT_EQ(e.entry_name, "name");
    EXPECT_EQ(e.entry_value, type{});
}

TYPED_TEST(TrackingEntry, output_operator) {
    using type = typename TestFixture::fixture_type;

    // construct a tracking entry
    const plssvm::detail::tracking::tracking_entry e{ "category", "name", type{} };

    // output the value
    std::cout << e;

    // check the output value
    EXPECT_EQ(this->get_capture(), fmt::format("{}", type{}));
}

TEST(TrackingEntry, is_tracking_entry) {
    // check whether the provided type is a tracking entry or not, ignoring any top-level const, reference, and volatile qualifiers
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry<plssvm::detail::tracking::tracking_entry<int>>::value);
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry_v<plssvm::detail::tracking::tracking_entry<int>>);
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry<plssvm::detail::tracking::tracking_entry<const int &>>::value);
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry_v<plssvm::detail::tracking::tracking_entry<const int &>>);
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry<plssvm::detail::tracking::tracking_entry<std::string>>::value);
    EXPECT_TRUE(plssvm::detail::tracking::is_tracking_entry_v<plssvm::detail::tracking::tracking_entry<std::string>>);
}

TEST(TrackingEntry, is_no_tracking_entry) {
    // the following types are NOT tracking entries
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry<int>::value);
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry_v<int>);
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry<const int &>::value);
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry_v<const int &>);
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry<std::string>::value);
    EXPECT_FALSE(plssvm::detail::tracking::is_tracking_entry_v<std::string>);
}

class PerformanceTracker : public ::testing::Test,
                           public util::redirect_output<&std::clog> {
  protected:
    void TearDown() override {
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
        // clear possible tracking entries stored from previous tests
        plssvm::detail::tracking::global_performance_tracker().clear_tracking_entries();
#endif
    }

    [[nodiscard]] plssvm::detail::tracking::performance_tracker &get_performance_tracker() noexcept {
        return tracker_;
    }

  private:
    plssvm::detail::tracking::performance_tracker tracker_{};
};

// the macros are only available if PLSSVM_PERFORMANCE_TRACKER_ENABLED is defined!
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

TEST_F(PerformanceTracker, pause_and_resume_macros) {
    // tracking is enabled per default
    EXPECT_TRUE(plssvm::detail::tracking::global_performance_tracker().is_tracking());
    // disable performance tracking
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_PAUSE();
    // tracking should now be disabled
    EXPECT_FALSE(plssvm::detail::tracking::global_performance_tracker().is_tracking());
    // re-enable performance tracking
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_RESUME();
    // tracking should now be enabled again
    EXPECT_TRUE(plssvm::detail::tracking::global_performance_tracker().is_tracking());
}

TEST_F(PerformanceTracker, save_macro) {
    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // read the file
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines();

    // test file contents
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("mem: 1024"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must not have changed
    EXPECT_EQ(plssvm::detail::tracking::global_performance_tracker().get_tracking_entries().size(), 2);

    // clear tracking entries for next test
    plssvm::detail::tracking::global_performance_tracker().clear_tracking_entries();
}

TEST_F(PerformanceTracker, set_reference_time_macro) {
    // set the new reference time
    const std::chrono::steady_clock::time_point ref_time = std::chrono::steady_clock::now();
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME(ref_time);

    // check if the reference time was correctly set
    EXPECT_EQ(plssvm::detail::tracking::global_performance_tracker().get_reference_time(), ref_time);
}

TEST_F(PerformanceTracker, add_entry_macro) {
    // add different tracking entries
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' }));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' }));

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::tracking::global_performance_tracker().get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 3);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "42");
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);
    EXPECT_EQ(entries.at("foo").at("baz").front(), "3.1415");
    ASSERT_EQ(entries.at("foo").at("mem").size(), 1);
    EXPECT_EQ(entries.at("foo").at("mem").front(), "1024");

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
    EXPECT_EQ(entries.at("").at("foobar"), (std::vector<std::string>{ "a", "b" }));

    // clear tracking entries for next test
    plssvm::detail::tracking::global_performance_tracker().clear_tracking_entries();
}

TEST_F(PerformanceTracker, add_hardware_sampler_entry_macro) {
    using namespace std::chrono_literals;

    // create the mocked hardware sampler
    const mock_hardware_sampler sampler1{ std::size_t{ 0 }, 50ms };
    const mock_hardware_sampler sampler2{ std::size_t{ 1 }, 100ms };

    EXPECT_CALL(sampler1, generate_yaml_string(::testing::An<std::chrono::steady_clock::time_point>())).Times(1);
    EXPECT_CALL(sampler1, device_identification()).Times(1);
    EXPECT_CALL(sampler2, generate_yaml_string(::testing::An<std::chrono::steady_clock::time_point>())).Times(1);
    EXPECT_CALL(sampler2, device_identification()).Times(1);

    // save the sampling entries
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY(sampler1);
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY(sampler2);

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = plssvm::detail::tracking::global_performance_tracker().get_tracking_entries();

    // check entries for correctness
    ASSERT_EQ(entries.size(), 1);

    EXPECT_EQ(entries.at("hardware_samples").size(), 2);
}

TEST_F(PerformanceTracker, add_event_macro) {
    // add different events
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("event_1");
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT("event_2");

    // get the events
    const plssvm::detail::tracking::events events = plssvm::detail::tracking::global_performance_tracker().get_events();

    // check entries for correctness
    ASSERT_EQ(events.num_events(), 2);

    // check event names
    EXPECT_EQ(events[0].name, std::string{ "event_1" });
    EXPECT_EQ(events[1].name, std::string{ "event_2" });

    // time points of the events should monotonically increase
    EXPECT_GE(events[1].time_point, events[0].time_point);
}

#endif

TEST_F(PerformanceTracker, pause_and_resume) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // tracking is enabled per default
    EXPECT_TRUE(tracker.is_tracking());
    // disable performance tracking
    tracker.pause_tracking();
    // tracking should now be disabled
    EXPECT_FALSE(tracker.is_tracking());
    // re-enable performance tracking
    tracker.resume_tracking();
    // tracking should now be enabled again
    EXPECT_TRUE(tracker.is_tracking());
}

TEST_F(PerformanceTracker, add_generic_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add different tracking entries
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 3);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "42");
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);
    EXPECT_EQ(entries.at("foo").at("baz").front(), "3.1415");
    ASSERT_EQ(entries.at("foo").at("mem").size(), 1);
    EXPECT_EQ(entries.at("foo").at("mem").front(), "1024");

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
    EXPECT_EQ(entries.at("").at("foobar"), (std::vector<std::string>{ "a", "b" }));
}

TEST_F(PerformanceTracker, add_string_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add a tracking entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", std::string{ "baz" } });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("foo").size(), 1);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    EXPECT_EQ(entries.at("foo").at("bar").front(), "\"baz\"");
}

TEST_F(PerformanceTracker, add_vector_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add a tracking entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", std::vector{ 1, 2, 3 } });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", std::vector{ std::string{ "a" }, std::string{ "b" } } });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("foo").size(), 1);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 2);
    EXPECT_EQ(entries.at("foo").at("bar"), (std::vector<std::string>{ "[1, 2, 3]", "[\"a\", \"b\"]" }));
}

TEST_F(PerformanceTracker, add_parameter_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add a tracking entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "parameter", "", plssvm::parameter{} });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 6);
}

TEST_F(PerformanceTracker, add_parser_train_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // create a parameter train object
    std::array<std::string, 3> input_argv{ "./plssvm-train", "/path/to/train", "/path/to/model" };
    std::array<char *, input_argv.size()> argv{};
    std::transform(input_argv.begin(), input_argv.end(), argv.begin(), [](std::string &str) { return str.data(); });

    const plssvm::detail::cmd::parser_train parser{ argv.size(), argv.data() };

    // save cmd::parser_train entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 17);
}

TEST_F(PerformanceTracker, add_parser_predict_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // create a parameter train object
    std::array<std::string, 4> input_argv{ "./plssvm-predict", "/path/to/train", "/path/to/model", "/path/to/predict" };
    std::array<char *, input_argv.size()> argv{};
    std::transform(input_argv.begin(), input_argv.end(), argv.begin(), [](std::string &str) { return str.data(); });

    const plssvm::detail::cmd::parser_predict parser{ argv.size(), argv.data() };

    // save cmd::parser_predict entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 9);
}

TEST_F(PerformanceTracker, add_parser_scale_tracking_entry) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // create a parameter train object
    std::array<std::string, 3> input_argv{ "./plssvm-train", "/path/to/train", "/path/to/scaled" };
    std::array<char *, input_argv.size()> argv{};
    std::transform(input_argv.begin(), input_argv.end(), argv.begin(), [](std::string &str) { return str.data(); });

    const plssvm::detail::cmd::parser_scale parser{ argv.size(), argv.data() };

    // save cmd::parser_scale entry
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "parameter", "", parser });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 1);

    ASSERT_EQ(entries.at("parameter").size(), 10);
}

TEST_F(PerformanceTracker, add_hardware_sampler_entry) {
    using namespace std::chrono_literals;

    // create the mocked hardware sampler
    const mock_hardware_sampler sampler1{ std::size_t{ 0 }, 50ms };
    const mock_hardware_sampler sampler2{ std::size_t{ 1 }, 100ms };

    EXPECT_CALL(sampler1, generate_yaml_string(::testing::An<std::chrono::steady_clock::time_point>())).Times(1);
    EXPECT_CALL(sampler1, device_identification()).Times(1);
    EXPECT_CALL(sampler2, generate_yaml_string(::testing::An<std::chrono::steady_clock::time_point>())).Times(1);
    EXPECT_CALL(sampler2, device_identification()).Times(1);

    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // save the sampling entries
    tracker.add_hardware_sampler_entry(sampler1);
    tracker.add_hardware_sampler_entry(sampler2);

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    ASSERT_EQ(entries.size(), 1);

    EXPECT_EQ(entries.at("hardware_samples").size(), 2);
}

TEST_F(PerformanceTracker, add_event) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add new event
    tracker.add_event("EVENT");

    // check for correctness
    ASSERT_EQ(tracker.get_events().num_events(), 1);
    EXPECT_EQ(tracker.get_events()[0].name, std::string{ "EVENT" });
}

TEST_F(PerformanceTracker, set_reference_time) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // set the new reference time
    const std::chrono::steady_clock::time_point ref_time = std::chrono::steady_clock::now();
    tracker.set_reference_time(ref_time);

    // check if the reference time was correctly set
    EXPECT_EQ(tracker.get_reference_time(), ref_time);
}

TEST_F(PerformanceTracker, get_reference_time) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // set the new reference time
    const std::chrono::steady_clock::time_point ref_time = std::chrono::steady_clock::now();
    tracker.set_reference_time(ref_time);

    // check if the reference time was correctly set
    EXPECT_EQ(tracker.get_reference_time(), ref_time);
}

TEST_F(PerformanceTracker, save_no_additional_entries) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    tracker.save(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // the tracking entries must be empty
    EXPECT_TRUE(tracker.get_tracking_entries().empty());
}

TEST_F(PerformanceTracker, save_entries_to_file) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // create temporary file
    const util::temporary_file tmp_file{};  // automatically removes the created file at the end of its scope
    // save entries to file
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });
    tracker.save(tmp_file.filename);

    // the file must not be empty
    EXPECT_FALSE(std::filesystem::is_empty(tmp_file.filename));

    // read the file
    plssvm::detail::io::file_reader reader{ tmp_file.filename };
    reader.read_lines('#');

    // test file contents
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("mem: 1024"));
    EXPECT_THAT(reader.buffer(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must not have changed
    EXPECT_EQ(tracker.get_tracking_entries().size(), 2);
}

TEST_F(PerformanceTracker, save_entries_empty_file) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // save entries to file
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });
    // save to empty file, i.e., dump the performance tracking entries to std::clog
    tracker.save("");

    // the captured standard output must not be empty
    EXPECT_FALSE(this->get_capture().empty());

    // test file contents
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("foo:"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("bar: 42"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("baz: 3.1415"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("mem: 1024"));
    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("foobar: [a, b]"));

    // the tracking entries must not have changed
    EXPECT_EQ(tracker.get_tracking_entries().size(), 2);
}

TEST_F(PerformanceTracker, get_tracking_entries) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add different tracking entries
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });

    // get the tracking entries
    const std::map<std::string, std::map<std::string, std::vector<std::string>>> entries = tracker.get_tracking_entries();

    // check entries for correctness
    EXPECT_EQ(entries.size(), 2);

    // first category
    ASSERT_EQ(entries.at("foo").size(), 3);
    ASSERT_EQ(entries.at("foo").at("bar").size(), 1);
    ASSERT_EQ(entries.at("foo").at("baz").size(), 1);
    ASSERT_EQ(entries.at("foo").at("mem").size(), 1);

    // second category
    ASSERT_EQ(entries.at("").size(), 1);
    ASSERT_EQ(entries.at("").at("foobar").size(), 2);
}

TEST_F(PerformanceTracker, get_events) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add different events
    tracker.add_event("event_1");
    tracker.add_event("event_2");
    tracker.add_event("event_3");

    // get the events
    const plssvm::detail::tracking::events events = tracker.get_events();

    // check for correctness
    ASSERT_EQ(events.num_events(), 3);

    // check event names
    EXPECT_EQ(events[0].name, std::string{ "event_1" });
    EXPECT_EQ(events[1].name, std::string{ "event_2" });
    EXPECT_EQ(events[2].name, std::string{ "event_3" });

    // time points of the events should monotonically increase
    EXPECT_GE(events[1].time_point, events[0].time_point);
    EXPECT_GE(events[2].time_point, events[1].time_point);
}

TEST_F(PerformanceTracker, clear_tracking_entries) {
    // get performance tracker from fixture class
    plssvm::detail::tracking::performance_tracker &tracker = this->get_performance_tracker();

    // add different tracking entries
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "bar", 42 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "baz", 3.1415 });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "foo", "mem", 1_KiB });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'a' });
    tracker.add_tracking_entry(plssvm::detail::tracking::tracking_entry{ "", "foobar", 'b' });

    // the performance tracker should contain entries
    EXPECT_FALSE(tracker.get_tracking_entries().empty());

    // clear all tracking entries
    tracker.clear_tracking_entries();

    // the performance tracker should now contain no tracking entries
    EXPECT_TRUE(tracker.get_tracking_entries().empty());
}
