/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the samples used with ROCm SMI.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint64_t, std::int64_t, std::uint32_t
#include <iosfwd>   // std::ostream forward declaration
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

class rocm_smi_general_samples {
  public:
    struct rocm_smi_general_sample {
        int performance_state{ 0 };
        std::uint32_t utilization_gpu{ 0 };
        std::uint32_t utilization_mem{ 0 };
    };

    explicit rocm_smi_general_samples(const std::uint32_t device_id) :
        device_id_{ device_id } { }

    std::string name{};

    void add_sample(rocm_smi_general_sample s) {
        this->performance_state_.push_back(s.performance_state);
        this->utilization_gpu_.push_back(s.utilization_gpu);
        this->utilization_mem_.push_back(s.utilization_mem);

        PLSSVM_ASSERT(this->num_samples() == this->performance_state_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->utilization_gpu_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->utilization_mem_.size(), "Error: number of general samples missmatch!");
    }

    rocm_smi_general_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return rocm_smi_general_sample{ performance_state_[idx], utilization_gpu_[idx], utilization_mem_[idx] };
    }

    [[nodiscard]] std::uint32_t get_device() const noexcept { return device_id_; }

    [[nodiscard]] std::size_t num_samples() const noexcept { return performance_state_.size(); }

    [[nodiscard]] bool empty() const noexcept { return performance_state_.empty(); }

    [[nodiscard]] const auto &get_performance_state() const noexcept { return performance_state_; }

    [[nodiscard]] const auto &get_utilization_gpu() const noexcept { return utilization_gpu_; }

    [[nodiscard]] const auto &get_utilization_mem() const noexcept { return utilization_mem_; }

  private:
    std::uint32_t device_id_;

    std::vector<decltype(rocm_smi_general_sample::performance_state)> performance_state_{};
    std::vector<decltype(rocm_smi_general_sample::utilization_gpu)> utilization_gpu_{};
    std::vector<decltype(rocm_smi_general_sample::utilization_mem)> utilization_mem_{};
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_general_samples &samples);

class rocm_smi_clock_samples {
  public:
    struct rocm_smi_clock_sample {
        std::uint64_t clock_system{ 0 };
        std::uint64_t clock_socket{ 0 };
        std::uint64_t clock_memory{ 0 };
        std::uint32_t clock_throttle_reason{ 0 };
        std::uint32_t overdrive_level{ 0 };
        std::uint32_t memory_overdrive_level{ 0 };
    };

    explicit rocm_smi_clock_samples(const std::uint32_t device_id) :
        device_id_{ device_id } { }

    std::uint64_t clock_system_min{ 0 };
    std::uint64_t clock_system_max{ 0 };
    std::uint64_t clock_socket_min{ 0 };
    std::uint64_t clock_socket_max{ 0 };
    std::uint64_t clock_memory_min{ 0 };
    std::uint64_t clock_memory_max{ 0 };

    void add_sample(rocm_smi_clock_sample s) {
        this->clock_system_.push_back(s.clock_system);
        this->clock_socket_.push_back(s.clock_socket);
        this->clock_memory_.push_back(s.clock_memory);
        this->clock_throttle_reason_.push_back(s.clock_throttle_reason);
        this->overdrive_level_.push_back(s.overdrive_level);
        this->memory_overdrive_level_.push_back(s.memory_overdrive_level);

        PLSSVM_ASSERT(this->num_samples() == this->clock_system_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_socket_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_memory_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->clock_throttle_reason_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->overdrive_level_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->memory_overdrive_level_.size(), "Error: number of general samples missmatch!");
    }

    rocm_smi_clock_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return rocm_smi_clock_sample{ clock_system_[idx], clock_socket_[idx], clock_memory_[idx], clock_throttle_reason_[idx], overdrive_level_[idx], memory_overdrive_level_[idx] };
    }

    [[nodiscard]] std::uint32_t get_device() const noexcept { return device_id_; }

    [[nodiscard]] std::size_t num_samples() const noexcept { return clock_system_.size(); }

    [[nodiscard]] bool empty() const noexcept { return clock_system_.empty(); }

    [[nodiscard]] const auto &get_clock_system() const noexcept { return clock_system_; }

    [[nodiscard]] const auto &get_clock_socket() const noexcept { return clock_socket_; }

    [[nodiscard]] const auto &get_clock_memory() const noexcept { return clock_memory_; }

    [[nodiscard]] const auto &get_clock_throttle_reason() const noexcept { return clock_throttle_reason_; }

    [[nodiscard]] const auto &get_overdrive_level() const noexcept { return overdrive_level_; }

    [[nodiscard]] const auto &get_memory_overdrive_level() const noexcept { return memory_overdrive_level_; }

  private:
    std::uint32_t device_id_;

    std::vector<decltype(rocm_smi_clock_sample::clock_system)> clock_system_{};
    std::vector<decltype(rocm_smi_clock_sample::clock_socket)> clock_socket_{};
    std::vector<decltype(rocm_smi_clock_sample::clock_memory)> clock_memory_{};
    std::vector<decltype(rocm_smi_clock_sample::clock_throttle_reason)> clock_throttle_reason_{};
    std::vector<decltype(rocm_smi_clock_sample::overdrive_level)> overdrive_level_{};
    std::vector<decltype(rocm_smi_clock_sample::memory_overdrive_level)> memory_overdrive_level_{};
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_clock_samples &samples);

class rocm_smi_power_samples {
  public:
    struct rocm_smi_power_sample {
        std::uint64_t power_usage{ 0 };                     // current power draw in W
        std::uint64_t power_total_energy_consumption{ 0 };  // total energy consumption since last driver reload in J
    };

    explicit rocm_smi_power_samples(const std::uint32_t device_id) :
        device_id_{ device_id } { }

    std::uint64_t power_default_cap{ 0 };  // maximum power limit in W
    std::uint64_t power_cap{ 0 };          // default power limit in W
    std::string power_type{};

    void add_sample(rocm_smi_power_sample s) {
        this->power_usage_.push_back(s.power_usage);
        this->power_total_energy_consumption_.push_back(s.power_total_energy_consumption);

        PLSSVM_ASSERT(this->num_samples() == this->power_usage_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->power_total_energy_consumption_.size(), "Error: number of general samples missmatch!");
    }

    rocm_smi_power_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return rocm_smi_power_sample{ power_usage_[idx], power_total_energy_consumption_[idx] };
    }

    [[nodiscard]] std::uint32_t get_device() const noexcept { return device_id_; }

    [[nodiscard]] std::size_t num_samples() const noexcept { return power_usage_.size(); }

    [[nodiscard]] bool empty() const noexcept { return power_usage_.empty(); }

    [[nodiscard]] const auto &get_power_usage() const noexcept { return power_usage_; }

    [[nodiscard]] const auto &get_power_total_energy_consumption() const noexcept { return power_total_energy_consumption_; }

  private:
    std::uint32_t device_id_;

    std::vector<decltype(rocm_smi_power_sample::power_usage)> power_usage_{};
    std::vector<decltype(rocm_smi_power_sample::power_total_energy_consumption)> power_total_energy_consumption_{};
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_power_samples &samples);

class rocm_smi_memory_samples {
  public:
    struct rocm_smi_memory_sample {
        std::uint64_t memory_used{ 0 };
        std::uint64_t pcie_transfer_rate{ 0 };
        std::uint32_t num_pcie_lanes{ 0 };
    };

    explicit rocm_smi_memory_samples(const std::uint32_t device_id) :
        device_id_{ device_id } { }

    std::uint64_t memory_total{ 0 };
    std::uint64_t visible_memory_total{ 0 };
    std::uint32_t min_num_pcie_lanes{ 0 };
    std::uint32_t max_num_pcie_lanes{ 0 };

    void add_sample(rocm_smi_memory_sample s) {
        this->memory_used_.push_back(s.memory_used);
        this->pcie_transfer_rate_.push_back(s.pcie_transfer_rate);
        this->num_pcie_lanes_.push_back(s.num_pcie_lanes);

        PLSSVM_ASSERT(this->num_samples() == this->memory_used_.size(), "Error: number of general samples missmatch!");
    }

    rocm_smi_memory_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return rocm_smi_memory_sample{ memory_used_[idx], pcie_transfer_rate_[idx], num_pcie_lanes_[idx] };
    }

    [[nodiscard]] std::uint32_t get_device() const noexcept { return device_id_; }

    [[nodiscard]] std::size_t num_samples() const noexcept { return memory_used_.size(); }

    [[nodiscard]] bool empty() const noexcept { return memory_used_.empty(); }

    [[nodiscard]] const auto &get_memory_used() const noexcept { return memory_used_; }

    [[nodiscard]] const auto &get_pcie_transfer_rate() const noexcept { return pcie_transfer_rate_; }

    [[nodiscard]] const auto &get_num_pcie_lanes() const noexcept { return num_pcie_lanes_; }

  private:
    std::uint32_t device_id_;

    std::vector<decltype(rocm_smi_memory_sample::memory_used)> memory_used_{};
    std::vector<decltype(rocm_smi_memory_sample::pcie_transfer_rate)> pcie_transfer_rate_{};
    std::vector<decltype(rocm_smi_memory_sample::num_pcie_lanes)> num_pcie_lanes_{};
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_memory_samples &samples);

class rocm_smi_temperature_samples {
  public:
    struct rocm_smi_temperature_sample {
        std::int64_t fan_speed{ 0 };
        std::int64_t temperature_edge{ 0 };
        std::int64_t temperature_hotspot{ 0 };
        std::int64_t temperature_memory{ 0 };
    };

    explicit rocm_smi_temperature_samples(const std::uint32_t device_id) :
        device_id_{ device_id } { }

    std::uint32_t num_fans{ 0 };
    std::uint64_t max_fan_speed{};
    std::int64_t temperature_edge_min{ 0 };
    std::int64_t temperature_edge_max{ 0 };
    std::int64_t temperature_hotspot_min{ 0 };
    std::int64_t temperature_hotspot_max{ 0 };
    std::int64_t temperature_memory_min{ 0 };
    std::int64_t temperature_memory_max{ 0 };

    void add_sample(rocm_smi_temperature_sample s) {
        this->fan_speed_.push_back(s.fan_speed);
        this->temperature_edge_.push_back(s.temperature_edge);
        this->temperature_hotspot_.push_back(s.temperature_hotspot);
        this->temperature_memory_.push_back(s.temperature_memory);

        PLSSVM_ASSERT(this->num_samples() == this->fan_speed_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->temperature_edge_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->temperature_hotspot_.size(), "Error: number of general samples missmatch!");
        PLSSVM_ASSERT(this->num_samples() == this->temperature_memory_.size(), "Error: number of general samples missmatch!");
    }

    rocm_smi_temperature_sample operator[](const std::size_t idx) const noexcept {
        PLSSVM_ASSERT(idx < this->num_samples(), "Error: out-of-bounce access with index {} for size {}!", idx, this->num_samples());

        return rocm_smi_temperature_sample{ fan_speed_[idx], temperature_edge_[idx], temperature_hotspot_[idx], temperature_memory_[idx] };
    }

    [[nodiscard]] std::uint32_t get_device() const noexcept { return device_id_; }

    [[nodiscard]] std::size_t num_samples() const noexcept { return fan_speed_.size(); }

    [[nodiscard]] bool empty() const noexcept { return fan_speed_.empty(); }

    [[nodiscard]] const auto &get_fan_speed() const noexcept { return fan_speed_; }

    [[nodiscard]] const auto &get_temperature_edge() const noexcept { return temperature_edge_; }

    [[nodiscard]] const auto &get_temperature_hotspot() const noexcept { return temperature_hotspot_; }

    [[nodiscard]] const auto &get_temperature_memory() const noexcept { return temperature_memory_; }

  private:
    std::uint32_t device_id_;

    std::vector<decltype(rocm_smi_temperature_sample::fan_speed)> fan_speed_{};
    std::vector<decltype(rocm_smi_temperature_sample::temperature_edge)> temperature_edge_{};
    std::vector<decltype(rocm_smi_temperature_sample::temperature_hotspot)> temperature_hotspot_{};
    std::vector<decltype(rocm_smi_temperature_sample::temperature_memory)> temperature_memory_{};
};

std::ostream &operator<<(std::ostream &out, const rocm_smi_temperature_samples &samples);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_general_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_clock_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_power_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_memory_samples> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::tracking::rocm_smi_temperature_samples> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_AMD_ROCM_SMI_SAMPLES_HPP_
