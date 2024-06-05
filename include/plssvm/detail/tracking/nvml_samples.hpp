/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with NVML.
 */

#ifndef PLSSVM_DETAIL_TRACKING_NVML_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_NVML_SAMPLES_HPP_

namespace plssvm::detail::tracking {

class nvml_general_samples {
  public:
    struct nvml_general_sample {
        std::chrono::milliseconds time_since_start{ 0 };
        int performance_state{ 0 };
        unsigned int utilization_gpu{ 0 };
        unsigned int utilization_mem{ 0 };
    };

    void add_sample(nvml_general_sample s) {
        this->times_since_start_.push_back(s.time_since_start);
        this->performance_states_.push_back(s.performance_state);
        this->utilizations_gpu_.push_back(s.utilization_gpu);
        this->utilizations_mem_.push_back(s.utilization_mem);

        // TODO: invariant!
    }

    nvml_general_sample operator[](const std::size_t idx) const noexcept {
        return nvml_general_sample{ times_since_start_[idx], performance_states_[idx], utilizations_gpu_[idx], utilizations_mem_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return times_since_start_.size(); }

    [[nodiscard]] bool empty() const noexcept { return times_since_start_.empty(); }

    [[nodiscard]] const auto &get_times_since_start() const noexcept { return times_since_start_; }

    [[nodiscard]] const auto &get_performance_states() const noexcept { return performance_states_; }

    [[nodiscard]] const auto &get_utilizations_gpu() const noexcept { return utilizations_gpu_; }

    [[nodiscard]] const auto &get_utilizations_mem() const noexcept { return utilizations_mem_; }

  private:
    std::vector<decltype(nvml_general_sample::time_since_start)> times_since_start_{};
    std::vector<decltype(nvml_general_sample::performance_state)> performance_states_{};
    std::vector<decltype(nvml_general_sample::utilization_gpu)> utilizations_gpu_{};
    std::vector<decltype(nvml_general_sample::utilization_mem)> utilizations_mem_{};
};

class nvml_clock_samples {
  public:
    struct nvml_clock_sample {
        unsigned int clock_graph{ 0 };
        unsigned int clock_sm{ 0 };
        unsigned int clock_mem{ 0 };
        unsigned long long clock_throttle_reason{ 0 };
        unsigned int clock_graph_max{ 0 };
        unsigned int clock_sm_max{ 0 };
        unsigned int clock_mem_max{ 0 };
    };

    void add_sample(nvml_clock_sample s) {
        this->clocks_graph_.push_back(s.clock_graph);
        this->clocks_sm_.push_back(s.clock_sm);
        this->clocks_mem_.push_back(s.clock_mem);
        this->clocks_throttle_reason_.push_back(s.clock_throttle_reason);
        this->clocks_graph_max_.push_back(s.clock_graph_max);
        this->clocks_sm_max_.push_back(s.clock_sm_max);
        this->clocks_mem_max_.push_back(s.clock_mem_max);

        // TODO: invariant!
    }

    nvml_clock_sample operator[](const std::size_t idx) const noexcept {
        return nvml_clock_sample{ clocks_graph_[idx], clocks_sm_[idx], clocks_mem_[idx], clocks_throttle_reason_[idx], clocks_graph_max_[idx], clocks_sm_max_[idx], clocks_mem_max_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return clocks_graph_.size(); }

    [[nodiscard]] bool empty() const noexcept { return clocks_graph_.empty(); }

    [[nodiscard]] const auto &get_clocks_graph() const noexcept { return clocks_graph_; }
    [[nodiscard]] const auto &get_clocks_sm() const noexcept { return clocks_sm_; }
    [[nodiscard]] const auto &get_clocks_mem() const noexcept { return clocks_mem_; }
    [[nodiscard]] const auto &get_clocks_throttle_reason() const noexcept { return clocks_throttle_reason_; }
    [[nodiscard]] const auto &get_clocks_graph_max() const noexcept { return clocks_graph_max_; }
    [[nodiscard]] const auto &get_clocks_sm_max() const noexcept { return clocks_sm_max_; }
    [[nodiscard]] const auto &get_clocks_mem_max() const noexcept { return clocks_mem_max_; }

  private:
    std::vector<decltype(nvml_clock_sample::clock_graph)> clocks_graph_{};
    std::vector<decltype(nvml_clock_sample::clock_sm)> clocks_sm_{};
    std::vector<decltype(nvml_clock_sample::clock_mem)> clocks_mem_{};
    std::vector<decltype(nvml_clock_sample::clock_throttle_reason)> clocks_throttle_reason_{};
    std::vector<decltype(nvml_clock_sample::clock_graph_max)> clocks_graph_max_{};
    std::vector<decltype(nvml_clock_sample::clock_sm_max)> clocks_sm_max_{};
    std::vector<decltype(nvml_clock_sample::clock_mem_max)> clocks_mem_max_{};
};

class nvml_temperature_samples {
  public:
    struct nvml_temperature_sample {
        unsigned int fan_speed{ 0 };
        unsigned int temperature_gpu{ 0 };
        unsigned int temperature_threshold_gpu_max{ 0 };
        unsigned int temperature_threshold_mem_max{ 0 };
    };

    void add_sample(nvml_temperature_sample s) {
        this->fan_speeds_.push_back(s.fan_speed);
        this->temperatures_gpu_.push_back(s.temperature_gpu);
        this->temperatures_threshold_gpu_max_.push_back(s.temperature_threshold_gpu_max);
        this->temperatures_threshold_mem_max_.push_back(s.temperature_threshold_mem_max);

        // TODO: invariant!
    }

    nvml_temperature_sample operator[](const std::size_t idx) const noexcept {
        return nvml_temperature_sample{ fan_speeds_[idx], temperatures_gpu_[idx], temperatures_threshold_gpu_max_[idx], temperatures_threshold_mem_max_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return fan_speeds_.size(); }

    [[nodiscard]] bool empty() const noexcept { return fan_speeds_.empty(); }

    [[nodiscard]] const auto &get_fan_speeds() const noexcept { return fan_speeds_; }

    [[nodiscard]] const auto &get_temperatures_gpu() const noexcept { return temperatures_gpu_; }

    [[nodiscard]] const auto &get_temperatures_threshold_gpu_max() const noexcept { return temperatures_threshold_gpu_max_; }

    [[nodiscard]] const auto &get_temperatures_threshold_mem_max() const noexcept { return temperatures_threshold_mem_max_; }

  private:
    std::vector<decltype(nvml_temperature_sample::fan_speed)> fan_speeds_{};
    std::vector<decltype(nvml_temperature_sample::temperature_gpu)> temperatures_gpu_{};
    std::vector<decltype(nvml_temperature_sample::temperature_threshold_gpu_max)> temperatures_threshold_gpu_max_{};
    std::vector<decltype(nvml_temperature_sample::temperature_threshold_mem_max)> temperatures_threshold_mem_max_{};
};

class nvml_memory_samples {
  public:
    struct nvml_memory_sample {
        unsigned long long memory_free{ 0 };
        unsigned long long memory_used{ 0 };
        unsigned long long memory_total{ 0 };
    };

    void add_sample(nvml_memory_sample s) {
        this->memory_free_.push_back(s.memory_free);
        this->memory_used_.push_back(s.memory_used);
        this->memory_total_.push_back(s.memory_total);

        // TODO: invariant!
    }

    nvml_memory_sample operator[](const std::size_t idx) const noexcept {
        return nvml_memory_sample{ memory_free_[idx], memory_used_[idx], memory_total_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return memory_free_.size(); }

    [[nodiscard]] bool empty() const noexcept { return memory_free_.empty(); }

    [[nodiscard]] const auto &get_memory_free() const noexcept { return memory_free_; }

    [[nodiscard]] const auto &get_memory_used() const noexcept { return memory_used_; }

    [[nodiscard]] const auto &get_memory_total() const noexcept { return memory_total_; }

  private:
    std::vector<decltype(nvml_memory_sample::memory_free)> memory_free_{};
    std::vector<decltype(nvml_memory_sample::memory_used)> memory_used_{};
    std::vector<decltype(nvml_memory_sample::memory_total)> memory_total_{};
};

class nvml_power_samples {
  public:
    struct nvml_power_sample {
        int power_state{ 0 };                                    // current power state
        unsigned int power_usage{ 0 };                           // current power draw in W
        unsigned int power_management_limit{ 0 };                // maximum power limit in W
        unsigned int power_enforced_limit{ 0 };                  // default power limit in W
        unsigned long long power_total_energy_consumption{ 0 };  // total energy consumption since last driver reload in J
    };

    void add_sample(nvml_power_sample s) {
        this->power_states_.push_back(s.power_state);
        this->power_usages_.push_back(s.power_usage);
        this->power_management_limits_.push_back(s.power_management_limit);
        this->power_enforced_limits_.push_back(s.power_enforced_limit);
        this->power_total_energy_consumptions_.push_back(s.power_total_energy_consumption);

        // TODO: invariant!
    }

    nvml_power_sample operator[](const std::size_t idx) const noexcept {
        return nvml_power_sample{ power_states_[idx], power_usages_[idx], power_management_limits_[idx], power_enforced_limits_[idx], power_total_energy_consumptions_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return power_states_.size(); }

    [[nodiscard]] bool empty() const noexcept { return power_states_.empty(); }

    [[nodiscard]] const auto &get_power_states() const noexcept { return power_states_; }

    [[nodiscard]] const auto &get_power_usages() const noexcept { return power_usages_; }

    [[nodiscard]] const auto &get_power_management_limits() const noexcept { return power_management_limits_; }

    [[nodiscard]] const auto &get_power_enforced_limits() const noexcept { return power_enforced_limits_; }

    [[nodiscard]] const auto &get_power_total_energy_consumptions() const noexcept { return power_total_energy_consumptions_; }

  private:
    std::vector<decltype(nvml_power_sample::power_state)> power_states_{};
    std::vector<decltype(nvml_power_sample::power_usage)> power_usages_{};
    std::vector<decltype(nvml_power_sample::power_management_limit)> power_management_limits_{};
    std::vector<decltype(nvml_power_sample::power_enforced_limit)> power_enforced_limits_{};
    std::vector<decltype(nvml_power_sample::power_total_energy_consumption)> power_total_energy_consumptions_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_NVML_SAMPLES_HPP_
