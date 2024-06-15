/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the performance tracker and hardware sampler.
 */

#ifndef PLSSVM_DETAIL_TRACKING_UTILITY_HPP_
#define PLSSVM_DETAIL_TRACKING_UTILITY_HPP_
#pragma once

#include <chrono>   // std::chrono::milliseconds
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

template <typename Duration = std::chrono::milliseconds, typename TimePoint>
[[nodiscard]] inline std::vector<Duration> durations_from_reference_time(const std::vector<TimePoint> &time_points, const TimePoint &reference) {
    std::vector<Duration> durations(time_points.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < durations.size(); ++i) {
        durations[i] = std::chrono::duration_cast<Duration>(time_points[i] - reference);
    }

    return durations;
}

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_UTILITY_HPP_
