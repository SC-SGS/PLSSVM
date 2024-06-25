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

/**
 * @brief Defines a public optional getter with name `get_sample_name` and a private optional member with name `sample_name_`.
 */
#define PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER(sample_type, sample_name)                      \
  public:                                                                                \
    [[nodiscard]] const std::optional<sample_type> &get_##sample_name() const noexcept { \
        return sample_name##_;                                                           \
    }                                                                                    \
                                                                                         \
  private:                                                                               \
    std::optional<sample_type> sample_name##_{};

/**
 * @brief Defines a public optional vector getter with name `get_sample_name` and a private optional vector member with name `sample_name_`.
 * @details Same as `PLSSVM_SAMPLE_STRUCT_FIXED_MEMBER` but per sample_name multiple values can be tracked.
 */
#define PLSSVM_SAMPLE_STRUCT_SAMPLING_MEMBER(sample_type, sample_name)                                \
  public:                                                                                             \
    [[nodiscard]] const std::optional<std::vector<sample_type>> &get_##sample_name() const noexcept { \
        return sample_name##_;                                                                        \
    }                                                                                                 \
                                                                                                      \
  private:                                                                                            \
    std::optional<std::vector<sample_type>> sample_name##_{};


/**
 * @brief Convert all time points to their duration passed since the @p reference time point.
 * @tparam Duration the duration type to return
 * @tparam TimePoint the type if the time points
 * @param[in] time_points the time points
 * @param[in] reference the reference time point
 * @return the duration passed since the @p reference time point (`[[nodiscard]]`)
 */
template <typename Duration = std::chrono::milliseconds, typename TimePoint>
[[nodiscard]] inline std::vector<Duration> durations_from_reference_time(const std::vector<TimePoint> &time_points, const TimePoint &reference) {
    std::vector<Duration> durations(time_points.size());

#pragma omp parallel for
    for (std::size_t i = 0; i < durations.size(); ++i) {
        durations[i] = std::chrono::duration_cast<Duration>(time_points[i] - reference);
    }

    return durations;
}

/**
 * @brief Convert all time points to their duration since the epoch start.
 * @tparam TimePoint the type of the time points
 * @param[in] time_points the time points
 * @return the duration passed since the respective @p TimePoint epoch start (`[[nodiscard]]`)
 */
template <typename TimePoint>
[[nodiscard]] inline std::vector<typename TimePoint::duration> time_points_to_epoch(const std::vector<TimePoint> &time_points) {
    std::vector<typename TimePoint::duration> times(time_points.size());
#pragma omp parallel for
    for (std::size_t i = 0; i < times.size(); ++i) {
        times[i] = time_points[i].time_since_epoch();
    }
    return times;
}

/**
 * @brief Return the value encapsulated by the std::optional @p opt if it contains a value, otherwise a default constructed @p T is returned.
 * @tparam T the type of the value stored in the std::optional
 * @param[in] opt the std::optional to check
 * @return the value of the std::optional or a default constructed @p T (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T value_or_default(const std::optional<T> &opt) {
    if (opt.has_value()) {
        return opt.value();
    } else {
        return T{};
    }
}

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_UTILITY_HPP_
