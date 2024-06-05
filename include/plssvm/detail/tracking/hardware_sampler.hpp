/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/event_type.hpp"  // plssvm::detail::tracking::event_type
#include "plssvm/detail/tracking/time_point.hpp"  // plssvm::detail::tracking::clock_type

#include <atomic>     // std::atomic
#include <stdexcept>  // std::runtime_error
#include <thread>     // std::thread
#include <utility>    // std::move
#include <vector>     // std::vector

#include <fstream>

namespace plssvm::detail::tracking {

template <typename T>
class hardware_sampler {
  public:
    using sample_type = T;

    explicit hardware_sampler(unsigned long long sampling_interval = 100ull);

    hardware_sampler(const hardware_sampler &) = delete;
    hardware_sampler(hardware_sampler &&) noexcept = default;
    hardware_sampler &operator=(const hardware_sampler &) = delete;
    hardware_sampler &operator=(hardware_sampler &&) noexcept = default;

    virtual ~hardware_sampler() = 0;

    void start_sampling();
    void stop_sampling();
    void pause_sampling();
    void resume_sampling();

    [[nodiscard]] bool is_sampling() const noexcept;

    // TODO: logic -> reset atomics to be able to completely restart sampling?
    void clear();
    void clear_samples();
    void clear_events();

    void add_event(event_type event);
    void add_event(std::string name);

    [[nodiscard]] std::size_t num_samples() const noexcept;
    [[nodiscard]] const sample_type &get_sample(std::size_t idx) const noexcept;
    [[nodiscard]] const std::vector<sample_type> &get_samples() const noexcept;

    [[nodiscard]] std::size_t num_events() const noexcept;
    [[nodiscard]] const event_type &get_event(std::size_t idx) const noexcept;
    [[nodiscard]] const std::vector<event_type> &get_events() const noexcept;

    // TODO: remove again
    void save(const std::string &filename) const {
        //
        std::ofstream out{ filename };
        // write header information
        out << "time,clock_graph,clock_sm,clock_mem,clock_throttle_reason,clock_graph_max,clock_sm_max,clock_mem_max,"
               "fan_speed,temperature_gpu,temperature_threshold_gpu_max,temperature_threshold_mem_max,memory_free,memory_used,memory_total,"
               "power_state,power_usage,power_limit,power_management_limit,power_consumption_total,performance_state,utilization_gpu,utilization_mem\n";
        for (const sample_type &sample : samples_) {
            // write timestamp
            out << static_cast<double>(time_point_to_epoch(sample.time) - start_time_) / 1000.0 << ","; // time in s

            // clock related information
            out << sample.clock_graph << ',';            // graphics clock speed in MHz
            out << sample.clock_sm << ',';               // SM clock speed in MHz
            out << sample.clock_mem << ',';              // memory clock speed in MHz
            out << sample.clock_throttle_reason << ',';  // clocks throttle reason
            out << sample.clock_graph_max << ',';        // maximum graphics clock speed in MHz
            out << sample.clock_sm_max << ',';           // maximum SM clock speed in MHz
            out << sample.clock_mem_max << ',';          // maximum memory clock speed in MHz

            // temperature related information
            out << sample.fan_speed << ',';                      // fan speed in %
            out << sample.temperature_gpu << ',';                // GPU temperature in °C
            out << sample.temperature_threshold_gpu_max << ',';  // maximum GPU temperature in °C
            out << sample.temperature_threshold_mem_max << ',';  // maximum memory temperature in °C

            // memory related information
            out << sample.memory_free << ',';   // free memory in GB
            out << sample.memory_used << ',';   // used memory in GB
            out << sample.memory_total << ',';  // total available memory in GB

            // power related information
            out << sample.power_state << ',';                     // current power state
            out << sample.power_usage << ',';                     // current power draw in W
            out << sample.power_limit << ',';                     // maximum power limit in W
            out << sample.power_default_limit << ',';             // default power limit in W
            out << sample.power_total_energy_consumption << ',';  // total energy consumption since profiling start in J

            // general information
            out << sample.performance_state << ',';  // current performance state
            out << sample.utilization_gpu << ',';    // current GPU utilization in %
            out << sample.utilization_mem << '\n';   // current memory utilization in %
        }
    }

  protected:
    void sampling_loop();

    virtual sample_type get_sample_measurement() = 0;
    virtual std::uint64_t get_total_energy_consumption() = 0;

    std::atomic<bool> sampling_started_{ false };
    std::atomic<bool> sampling_stopped_{ false };
    std::atomic<bool> sampling_running_{ false };

    unsigned long long sampling_interval_;
    unsigned long long start_time_;
    std::thread sampling_thread_{};

    std::vector<sample_type> samples_;
    std::vector<event_type> events_;
    std::uint64_t power_total_energy_consumption_start_;
};

template <typename T>
hardware_sampler<T>::hardware_sampler(unsigned long long sampling_interval) :
    sampling_interval_{ sampling_interval } { }

template <typename T>
hardware_sampler<T>::~hardware_sampler() = default;

// TODO: use correct exception

template <typename T>
void hardware_sampler<T>::start_sampling() {
    // can't start an already running sampler
    if (sampling_started_) {
        throw std::runtime_error{ "Can't start an already running sampler!" };
    }

    // start sampling loop
    sampling_started_ = true;
    this->resume_sampling();
    start_time_ = time_point_to_epoch(clock_type::now());
    sampling_thread_ = std::thread{ [this]() { this->sampling_loop(); } };
    sampling_thread_.detach();
}

template <typename T>
void hardware_sampler<T>::stop_sampling() {
    // can't stop an already stopped sampler
    if (sampling_stopped_) {
        throw std::runtime_error{ "Can't stop an already stopped sampler!" };
    }

    // stop sampling
    this->pause_sampling();
    sampling_stopped_ = true;  // -> notifies the sampling std::thread
    if (sampling_thread_.joinable()) {
        sampling_thread_.join();
    }
}

template <typename T>
void hardware_sampler<T>::pause_sampling() {
    sampling_running_ = false;  // notifies the sampling std::thread
}

template <typename T>
void hardware_sampler<T>::resume_sampling() {
    sampling_running_ = true;  // notifies the sampling std::thread
}

template <typename T>
bool hardware_sampler<T>::is_sampling() const noexcept {
    return sampling_running_;
}

template <typename T>
void hardware_sampler<T>::clear() {
    this->clear_samples();
    this->clear_events();
}

template <typename T>
void hardware_sampler<T>::clear_samples() {
    samples_.clear();
}

template <typename T>
void hardware_sampler<T>::clear_events() {
    events_.clear();
}

template <typename T>
void hardware_sampler<T>::add_event(event_type event) {
    events_.push_back(std::move(event));
}

template <typename T>
void hardware_sampler<T>::add_event(std::string name) {
    events_.emplace_back(clock_type::now(), std::move(name));
}

template <typename T>
std::size_t hardware_sampler<T>::num_samples() const noexcept {
    return samples_.size();
}

template <typename T>
auto hardware_sampler<T>::get_sample(const std::size_t idx) const noexcept -> const sample_type & {
    // TODO: assert!
    return samples_[idx];
}

template <typename T>
auto hardware_sampler<T>::get_samples() const noexcept -> const std::vector<sample_type> & {
    return samples_;
}

template <typename T>
std::size_t hardware_sampler<T>::num_events() const noexcept {
    return events_.size();
}

template <typename T>
const event_type &hardware_sampler<T>::get_event(const std::size_t idx) const noexcept {
    // TODO: assert!
    return events_[idx];
}

template <typename T>
const std::vector<event_type> &hardware_sampler<T>::get_events() const noexcept {
    return events_;
}

template <typename T>
void hardware_sampler<T>::sampling_loop() {
    power_total_energy_consumption_start_ = this->get_total_energy_consumption();

    // loop until stop_sampling() is called
    while (!sampling_stopped_) {
        samples_.push_back(this->get_sample_measurement());

        // wait for sampling_interval_ milliseconds to retrieve the next sample
        std::this_thread::sleep_for(std::chrono::milliseconds{ sampling_interval_ });
    }
}

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
