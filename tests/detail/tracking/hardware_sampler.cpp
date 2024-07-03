/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the hardware sampler base class.
 */

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::hardware_sampler_exception

#include "tests/custom_test_macros.hpp"                     // EXPECT_THROW_WHAT
#include "tests/detail/tracking/mock_hardware_sampler.hpp"  // mock_hardware_sampler

#include "gmock/gmock.h"  // EXPECT_CALL
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ::testing::Test

#include <chrono>   // std::chrono::steady_clock::{time_point, now}, std::chrono::milliseconds
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

TEST(HardwareSampler, construct) {
    // construct a mock hardware sampler
    const mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // the sampler should not be running right after construction
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // the sampling interval should have been set correctly
    EXPECT_EQ(sampler.sampling_interval(), std::chrono::milliseconds{ 50 });
}

TEST(HardwareSampler, start_sampling) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // check values before call to start_sampling()
    EXPECT_FALSE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();

    // check values after call to start_sampling
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_TRUE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());
}

TEST(HardwareSampler, start_sampling_twice) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();

    // try start sampling again
    EXPECT_THROW_WHAT(sampler.start_sampling(), plssvm::hardware_sampling_exception, "Can start every hardware sampler only once!");
}

TEST(HardwareSampler, stop_sampling) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();

    // check values before call to stop_sampling()
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_TRUE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // stop sampling
    sampler.stop_sampling();

    // check values before call to stop_sampling()
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_TRUE(sampler.has_sampling_stopped());
}

TEST(HardwareSampler, stop_sampling_twice) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();

    // stop sampling
    sampler.stop_sampling();

    // try start sampling again
    EXPECT_THROW_WHAT(sampler.stop_sampling(), plssvm::hardware_sampling_exception, "Can stop every hardware sampler only once!");
}

TEST(HardwareSampler, stop_sampling_without_start) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // stop sampling without a call to start
    EXPECT_THROW_WHAT(sampler.stop_sampling(), plssvm::hardware_sampling_exception, "Can't stop a hardware sampler that has never been started!");
}

TEST(HardwareSampler, pause_sampling) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();

    // sampler is sampling
    EXPECT_TRUE(sampler.is_sampling());

    // pause sampling
    sampler.pause_sampling();

    // now, the sampler shouldn't be sampling!
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // a second call to pause shouldn't change anything
    sampler.pause_sampling();
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_FALSE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());
}

TEST(HardwareSampler, resume_sampling) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling and immediately pause it
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();
    sampler.pause_sampling();

    // sampler is NOT sampling
    EXPECT_FALSE(sampler.is_sampling());

    // resume sampling
    sampler.resume_sampling();

    // now, the sampler should again be sampling!
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_TRUE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // a second call to resume shouldn't change anything
    sampler.resume_sampling();
    EXPECT_TRUE(sampler.has_sampling_started());
    EXPECT_TRUE(sampler.is_sampling());
    EXPECT_FALSE(sampler.has_sampling_stopped());
}

TEST(HardwareSampler, resume_sampling_after_stopped) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // start sampling and immediately pause it
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();
    sampler.stop_sampling();

    // calling resume on a stopped hardware sampler should throw an exception
    EXPECT_THROW_WHAT(sampler.resume_sampling(), plssvm::hardware_sampling_exception, "Can't resume a hardware sampler that has already been stopped!");
}

TEST(HardwareSampler, has_sampling_started) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // sampling not started yet
    EXPECT_FALSE(sampler.has_sampling_started());

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();
    EXPECT_TRUE(sampler.has_sampling_started());

    // after stop_sampling -> has_sampling_started should still return true!
    sampler.stop_sampling();
    EXPECT_TRUE(sampler.has_sampling_started());
}

TEST(HardwareSampler, is_sampling) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // currently not sampling
    EXPECT_FALSE(sampler.is_sampling());

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();
    EXPECT_TRUE(sampler.is_sampling());

    // pause sampling
    sampler.pause_sampling();
    EXPECT_FALSE(sampler.is_sampling());

    // resume sampling
    sampler.resume_sampling();
    EXPECT_TRUE(sampler.is_sampling());

    // stop sampling
    sampler.stop_sampling();
    EXPECT_FALSE(sampler.is_sampling());
}

TEST(HardwareSampler, has_sampling_stopped) {
    // create mock hardware sampler
    mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // sampling not started yet
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // start sampling
    EXPECT_CALL(sampler, sampling_loop).Times(1);
    sampler.start_sampling();
    EXPECT_FALSE(sampler.has_sampling_stopped());

    // after stop_sampling -> has_sampling_stopped should return true
    sampler.stop_sampling();
    EXPECT_TRUE(sampler.has_sampling_stopped());
}

TEST(HardwareSampler, time_points) {
    // create mock hardware sampler
    const mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // return the time points, should be empty for the mock class
    EXPECT_TRUE(sampler.time_points().empty());
}

TEST(HardwareSampler, sampling_interval) {
    // create mock hardware sampler
    const mock_hardware_sampler sampler{ std::size_t{ 0 }, std::chrono::milliseconds{ 50 } };

    // return the sampling interval
    EXPECT_EQ(sampler.sampling_interval(), std::chrono::milliseconds{ 50 });
}
