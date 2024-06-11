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

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstddef>  // std::size_t
#include <iosfwd>   // std::ostream forward declaration
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

class nvml_general_samples {
  public:
    struct nvml_general_sample {
        int performance_state{ 0 };
        unsigned int utilization_gpu{ 0 };
        unsigned int utilization_mem{ 0 };
    };

    std::string name{};
    bool persistence_mode{ false };
    unsigned int num_cores{ 0 };

    void add_sample(nvml_general_sample s) {
        this->performance_state_.push_back(s.performance_state);
        this->utilization_gpu_.push_back(s.utilization_gpu);
        this->utilization_mem_.push_back(s.utilization_mem);

        PLSSVM_ASSERT(this->num_samples() == this->performance_state_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->utilization_gpu_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->utilization_mem_.size(), "Error: number of general samples missmatch!");
    }

    nvml_general_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return nvml_general_sample{ performance_state_[idx], utilization_gpu_[idx], utilization_mem_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return performance_state_.size(); }

    [[nodiscard]] bool empty() const noexcept { return performance_state_.empty(); }

    [[nodiscard]] const auto &get_performance_state() const noexcept { return performance_state_; }

    [[nodiscard]] const auto &get_utilization_gpu() const noexcept { return utilization_gpu_; }

    [[nodiscard]] const auto &get_utilization_mem() const noexcept { return utilization_mem_; }

  private:
    std::vector<decltype(nvml_general_sample::performance_state)> performance_state_{};
    std::vector<decltype(nvml_general_sample::utilization_gpu)> utilization_gpu_{};
    std::vector<decltype(nvml_general_sample::utilization_mem)> utilization_mem_{};
};

std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples);

class nvml_clock_samples {
  public:
    struct nvml_clock_sample {
        unsigned int clock_graph{ 0 };
        unsigned int clock_sm{ 0 };
        unsigned int clock_mem{ 0 };
        unsigned long long clock_throttle_reason{ 0 };
        bool auto_boosted_clocks{ false };
    };

    unsigned int adaptive_clock_status{ 0 };
    unsigned int clock_graph_max{ 0 };
    unsigned int clock_sm_max{ 0 };
    unsigned int clock_mem_max{ 0 };

    void add_sample(nvml_clock_sample s) {
        this->clock_graph_.push_back(s.clock_graph);
        this->clock_sm_.push_back(s.clock_sm);
        this->clock_mem_.push_back(s.clock_mem);
        this->clock_throttle_reason_.push_back(s.clock_throttle_reason);
        this->auto_boosted_clocks_.push_back(s.auto_boosted_clocks);

        PLSSVM_ASSERT(this->num_samples() == this->clock_graph_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_sm_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_mem_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_throttle_reason_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->auto_boosted_clocks_.size(), "Error: number of general samples missmatch!");
    }

    nvml_clock_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return nvml_clock_sample{ clock_graph_[idx], clock_sm_[idx], clock_mem_[idx], clock_throttle_reason_[idx], auto_boosted_clocks_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return clock_graph_.size(); }

    [[nodiscard]] bool empty() const noexcept { return clock_graph_.empty(); }

    [[nodiscard]] const auto &get_clock_graph() const noexcept { return clock_graph_; }

    [[nodiscard]] const auto &get_clock_sm() const noexcept { return clock_sm_; }

    [[nodiscard]] const auto &get_clock_mem() const noexcept { return clock_mem_; }

    [[nodiscard]] const auto &get_clock_throttle_reason() const noexcept { return clock_throttle_reason_; }

    [[nodiscard]] const auto &get_auto_boosted_clocks() const noexcept { return auto_boosted_clocks_; }

  private:
    std::vector<decltype(nvml_clock_sample::clock_graph)> clock_graph_{};
    std::vector<decltype(nvml_clock_sample::clock_sm)> clock_sm_{};
    std::vector<decltype(nvml_clock_sample::clock_mem)> clock_mem_{};
    std::vector<decltype(nvml_clock_sample::clock_throttle_reason)> clock_throttle_reason_{};
    std::vector<decltype(nvml_clock_sample::auto_boosted_clocks)> auto_boosted_clocks_{};
};

std::ostream &operator<<(std::ostream &out, const nvml_clock_samples &samples);

class nvml_power_samples {
  public:
    struct nvml_power_sample {
        int power_state{ 0 };                                    // current power state
        unsigned int power_usage{ 0 };                           // current power draw in W
        unsigned long long power_total_energy_consumption{ 0 };  // total energy consumption since last driver reload in J
    };

    unsigned int power_management_limit{ 0 };  // maximum power limit in W
    unsigned int power_enforced_limit{ 0 };    // default power limit in W

    void add_sample(nvml_power_sample s) {
        this->power_state_.push_back(s.power_state);
        this->power_usage_.push_back(s.power_usage);
        this->power_total_energy_consumption_.push_back(s.power_total_energy_consumption);

        PLSSVM_ASSERT(this->num_samples() == this->power_state_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->power_usage_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->power_total_energy_consumption_.size(), "Error: number of general samples missmatch!");
    }

    nvml_power_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return nvml_power_sample{ power_state_[idx], power_usage_[idx], power_total_energy_consumption_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return power_state_.size(); }

    [[nodiscard]] bool empty() const noexcept { return power_state_.empty(); }

    [[nodiscard]] const auto &get_power_state() const noexcept { return power_state_; }

    [[nodiscard]] const auto &get_power_usage() const noexcept { return power_usage_; }

    [[nodiscard]] const auto &get_power_total_energy_consumption() const noexcept { return power_total_energy_consumption_; }

  private:
    std::vector<decltype(nvml_power_sample::power_state)> power_state_{};
    std::vector<decltype(nvml_power_sample::power_usage)> power_usage_{};
    std::vector<decltype(nvml_power_sample::power_total_energy_consumption)> power_total_energy_consumption_{};
};

std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples);

class nvml_memory_samples {
  public:
    struct nvml_memory_sample {
        unsigned long long memory_free{ 0 };
        unsigned long long memory_used{ 0 };
        unsigned int pcie_link_width{ 0 };
        unsigned int pcie_link_generation{ 0 };
    };

    unsigned long long memory_total{ 0 };
    unsigned int memory_bus_width{ 0 };
    unsigned int max_pcie_link_generation{ 0 };
    unsigned int pcie_link_max_speed{ 0 };

    void add_sample(nvml_memory_sample s) {
        this->memory_free_.push_back(s.memory_free);
        this->memory_used_.push_back(s.memory_used);
        this->pcie_link_width_.push_back(s.pcie_link_width);
        this->pcie_link_generation_.push_back(s.pcie_link_generation);

        PLSSVM_ASSERT(this->num_samples() == this->memory_free_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->memory_used_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->pcie_link_width_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->pcie_link_generation_.size(), "Error: number of general samples missmatch!");
    }

    nvml_memory_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return nvml_memory_sample{ memory_free_[idx], memory_used_[idx], pcie_link_width_[idx], pcie_link_generation_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return memory_free_.size(); }

    [[nodiscard]] bool empty() const noexcept { return memory_free_.empty(); }

    [[nodiscard]] const auto &get_memory_free() const noexcept { return memory_free_; }

    [[nodiscard]] const auto &get_memory_used() const noexcept { return memory_used_; }

    [[nodiscard]] const auto &get_pcie_link_width() const noexcept { return pcie_link_width_; }

    [[nodiscard]] const auto &get_pcie_link_generation() const noexcept { return pcie_link_generation_; }

  private:
    std::vector<decltype(nvml_memory_sample::memory_free)> memory_free_{};
    std::vector<decltype(nvml_memory_sample::memory_used)> memory_used_{};
    std::vector<decltype(nvml_memory_sample::pcie_link_width)> pcie_link_width_{};
    std::vector<decltype(nvml_memory_sample::pcie_link_generation)> pcie_link_generation_{};
};

std::ostream &operator<<(std::ostream &out, const nvml_memory_samples &samples);

class nvml_temperature_samples {
  public:
    struct nvml_temperature_sample {
        unsigned int fan_speed{ 0 };
        unsigned int temperature_gpu{ 0 };
    };

    unsigned int num_fans{ 0 };
    unsigned int min_fan_speed{ 0 };
    unsigned int max_fan_speed{ 0 };
    unsigned int temperature_threshold_gpu_max{ 0 };
    unsigned int temperature_threshold_mem_max{ 0 };

    void add_sample(nvml_temperature_sample s) {
        this->fan_speed_.push_back(s.fan_speed);
        this->temperature_gpu_.push_back(s.temperature_gpu);

        PLSSVM_ASSERT(this->num_samples() == this->fan_speed_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->temperature_gpu_.size(), "Error: number of general samples missmatch!");
    }

    nvml_temperature_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return nvml_temperature_sample{ fan_speed_[idx], temperature_gpu_[idx] };
    }

    [[nodiscard]] std::size_t num_samples() const noexcept { return fan_speed_.size(); }

    [[nodiscard]] bool empty() const noexcept { return fan_speed_.empty(); }

    [[nodiscard]] const auto &get_fan_speed() const noexcept { return fan_speed_; }

    [[nodiscard]] const auto &get_temperature_gpu() const noexcept { return temperature_gpu_; }

  private:
    std::vector<decltype(nvml_temperature_sample::fan_speed)> fan_speed_{};
    std::vector<decltype(nvml_temperature_sample::temperature_gpu)> temperature_gpu_{};
};

std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::nvml_temperature_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_NVML_SAMPLES_HPP_
