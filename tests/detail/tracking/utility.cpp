/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the tracking specific utility classes.
 */

#include "plssvm/detail/tracking/utility.hpp"

#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ, ::testing::StaticAssertTypeEq

#include <chrono>       // std::chrono::steady_clock::{time_point, now}, std::chrono::milliseconds
#include <cstddef>      // std::size_t
#include <optional>     // std::optional, std::nullopt
#include <string>       // std::string
#include <type_traits>  // std::result_of_t
#include <vector>       // std::vector

TEST(TrackingUtility, durations_from_reference_time) {
    // create different time points
    std::vector<std::chrono::steady_clock::time_point> time_points{ std::chrono::steady_clock::now() };
    for (std::size_t i = 0; i < 3; ++i) {
        time_points.push_back(time_points.front() + std::chrono::milliseconds{ (i + 1) * 50 });
    }

    // convert the time points to durations based on the reference time point
    const std::vector<std::chrono::milliseconds> durations = plssvm::detail::tracking::durations_from_reference_time(time_points, time_points.front());

    // check the durations
    using namespace std::chrono_literals;
    ASSERT_EQ(durations.size(), 4);
    EXPECT_EQ(durations, (std::vector<std::chrono::milliseconds>{ 0ms, 50ms, 100ms, 150ms }));
}

TEST(TrackingUtility, time_points_to_epoch) {
    // create different time points
    std::vector<std::chrono::steady_clock::time_point> time_points{ std::chrono::steady_clock::now() };
    for (std::size_t i = 0; i < 3; ++i) {
        time_points.push_back(time_points.front() + std::chrono::milliseconds{ (i + 1) * 50 });
    }

    // convert time points to epochs
    const std::vector<std::chrono::steady_clock::duration> epochs = plssvm::detail::tracking::time_points_to_epoch(time_points);

    ASSERT_EQ(epochs.size(), time_points.size());
    for (std::size_t i = 0; i < time_points.size(); ++i) {
        EXPECT_EQ(epochs[i], time_points[i].time_since_epoch());
    }
}
