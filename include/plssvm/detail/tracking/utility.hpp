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

#include <chrono>    // std::chrono::milliseconds
#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

#define PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(sample_type, sample_name)                      \
  public:                                                                                \
    [[nodiscard]] const std::optional<sample_type> &get_##sample_name() const noexcept { \
        return sample_name##_;                                                           \
    }                                                                                    \
                                                                                         \
  private:                                                                               \
    std::optional<sample_type> sample_name##_{};

#define PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(sample_type, sample_name)                                \
  public:                                                                                             \
    [[nodiscard]] const std::optional<std::vector<sample_type>> &get_##sample_name() const noexcept { \
        return sample_name##_;                                                                        \
    }                                                                                                 \
                                                                                                      \
  private:                                                                                            \
    std::optional<std::vector<sample_type>> sample_name##_{};

template <typename Duration = std::chrono::milliseconds, typename TimePoint>
[[nodiscard]] inline std::vector<Duration> durations_from_reference_time(const std::vector<TimePoint> &time_points, const TimePoint &reference) {
    std::vector<Duration> durations(time_points.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < durations.size(); ++i) {
        durations[i] = std::chrono::duration_cast<Duration>(time_points[i] - reference);
    }

    return durations;
}

template <typename T>
[[nodiscard]] T value_or_default(const std::optional<T> &opt) {
    if (opt.has_value()) {
        return opt.value();
    } else {
        return T{};
    }
}

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_UTILITY_HPP_
