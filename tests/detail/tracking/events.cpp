/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the events class and event struct.
 */

#include "plssvm/detail/tracking/events.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::redirect_output

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST_F, ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH, ::testing::Test

#include <chrono>    // std::chrono::steady_clock::{time_point, now}, std::chrono::milliseconds
#include <iostream>  // std::cout
#include <string>    // std::string
#include <tuple>     // std::ignore
#include <vector>    // std::vector

class Event : public ::testing::Test,
              protected util::redirect_output<> { };

TEST_F(Event, construct) {
    const std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

    // construct an event
    const plssvm::detail::tracking::events::event e{ time, "EVENT" };

    // check member variables
    EXPECT_EQ(e.time_point, time);
    EXPECT_EQ(e.name, std::string{ "EVENT" });
}

TEST_F(Event, output_operator) {
    const std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

    // construct an event
    const plssvm::detail::tracking::events::event e{ time, "EVENT" };

    // output the event
    std::cout << e;

    // get the output and check the values
    const std::string str = this->get_capture();
    EXPECT_EQ(str, fmt::format("time_point: {}\nname: EVENT", time.time_since_epoch()));
}

class Events : public ::testing::Test,
               protected util::redirect_output<&std::cout> { };

TEST_F(Events, construct) {
    // default construct an events wrapper
    const plssvm::detail::tracking::events events{};

    // must be empty!
    EXPECT_TRUE(events.empty());
}

TEST_F(Events, add_event) {
    // create events wrapper
    plssvm::detail::tracking::events events{};

    // create event
    const plssvm::detail::tracking::events::event e{ std::chrono::steady_clock::now(), "EVENT" };

    // add event to wrapper
    events.add_event(e);

    // check events
    ASSERT_EQ(events.num_events(), 1);

    EXPECT_EQ(events[0].time_point, e.time_point);
    EXPECT_EQ(events[0].name, e.name);
}

TEST_F(Events, add_event_with_time_point_and_name) {
    // create events wrapper
    plssvm::detail::tracking::events events{};

    // create event data
    const std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

    // add event to wrapper
    events.add_event(time, "EVENT");

    // check events
    ASSERT_EQ(events.num_events(), 1);

    EXPECT_EQ(events[0].time_point, time);
    EXPECT_EQ(events[0].name, std::string{ "EVENT" });
}

TEST_F(Events, get_event_by_index) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // check content using the operator[]
    ASSERT_EQ(events.num_events(), 2);

    EXPECT_EQ(events[0].time_point, time1);
    EXPECT_EQ(events[0].name, std::string{ "EVENT_1" });
    EXPECT_EQ(events[1].time_point, time2);
    EXPECT_EQ(events[1].name, std::string{ "EVENT_2" });
}

TEST_F(Events, get_event_at_index) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // check content using the at() function
    ASSERT_EQ(events.num_events(), 2);

    EXPECT_EQ(events.at(0).time_point, time1);
    EXPECT_EQ(events.at(0).name, std::string{ "EVENT_1" });
    EXPECT_EQ(events.at(1).time_point, time2);
    EXPECT_EQ(events.at(1).name, std::string{ "EVENT_2" });
}

TEST_F(Events, get_event_at_index_out_of_bounce) {
    // create events wrapper
    const plssvm::detail::tracking::events events{};

    EXPECT_THROW_WHAT(std::ignore = events.at(0), plssvm::exception, "Index 0 is out-of-bounce for the number of events 0!");
}

TEST_F(Events, num_events) {
    // create events wrapper
    plssvm::detail::tracking::events events{};

    // no events should currently be present
    EXPECT_EQ(events.num_events(), 0);

    // add an event
    events.add_event(std::chrono::steady_clock::now(), "EVENT");

    // one event should currently be present
    EXPECT_EQ(events.num_events(), 1);
}

TEST_F(Events, empty) {
    // create events wrapper
    plssvm::detail::tracking::events events{};

    // no events should currently be present
    EXPECT_TRUE(events.empty());

    // add an event
    events.add_event(std::chrono::steady_clock::now(), "EVENT");

    // one event should currently be present
    EXPECT_FALSE(events.empty());
}

TEST_F(Events, get_time_points) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // check the time points
    ASSERT_EQ(events.num_events(), events.get_time_points().size());
    EXPECT_EQ(events.get_time_points(), (std::vector<std::chrono::steady_clock::time_point>{ time1, time2 }));
}

TEST_F(Events, get_names) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // check the time points
    ASSERT_EQ(events.num_events(), events.get_names().size());
    EXPECT_EQ(events.get_names(), (std::vector<std::string>{ "EVENT_1", "EVENT_2" }));
}

TEST_F(Events, generate_yaml_string) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // get the YAML string
    const std::string yaml = events.generate_yaml_string(time1);

    // assemble the correct string
    const std::string correct_yaml = fmt::format("    time_points: [{}, {}]\n    names: [\"EVENT_1\", \"EVENT_2\"]",
                                                 std::chrono::milliseconds{ 0 },
                                                 std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1));

    // check for equality
    EXPECT_EQ(yaml, correct_yaml);
}

TEST_F(Events, output_operator) {
    const std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point time2 = std::chrono::steady_clock::now();
    // create events wrapper and add events
    plssvm::detail::tracking::events events{};
    events.add_event(time1, "EVENT_1");
    events.add_event(time2, "EVENT_2");

    // output the events
    std::cout << events;

    // get the output and check the values
    const std::string str = this->get_capture();
    EXPECT_EQ(str, fmt::format("time_points: [{}, {}]\nnames: [EVENT_1, EVENT_2]", time1.time_since_epoch(), time2.time_since_epoch()));
}

class EventsDeathTest : public Events { };

TEST_F(EventsDeathTest, get_event_by_index_out_of_bounce) {
    // create events wrapper
    const plssvm::detail::tracking::events events{};

    EXPECT_DEATH(std::ignore = events[0], "Index 0 is out-of-bounce for the number of events 0!");
}
