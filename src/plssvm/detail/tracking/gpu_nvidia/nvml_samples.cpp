/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_nvidia/nvml_samples.hpp"

#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::value_or_default

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join
#include "nvml.h"        // NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED

#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <ostream>   // std::ostream
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string nvml_general_samples::generate_yaml_string() const {
    std::string str{ "    general:\n" };

    // device name
    if (this->name_.has_value()) {
        str += fmt::format("      name:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->name_.value());
    }
    // persistence mode enabled
    if (this->persistence_mode_.has_value()) {
        str += fmt::format("      persistence_mode:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->persistence_mode_.value());
    }
    // number of cores
    if (this->num_cores_.has_value()) {
        str += fmt::format("      num_cores:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_cores_.value());
    }

    // performance state
    if (this->performance_state_.has_value()) {
        str += fmt::format("      performance_state:\n"
                           "        unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->performance_state_.value(), ", "));
    }
    // device compute utilization
    if (this->utilization_gpu_.has_value()) {
        str += fmt::format("      utilization_gpu:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->utilization_gpu_.value(), ", "));
    }

    // device compute utilization
    if (this->utilization_mem_.has_value()) {
        str += fmt::format("      utilization_mem:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->utilization_mem_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples) {
    return out << fmt::format("name: {}\n"
                              "persistence_mode: {}\n"
                              "num_cores: {}\n"
                              "performance_state: [{}]\n"
                              "utilization_gpu: [{}]\n"
                              "utilization_mem: [{}]",
                              value_or_default(samples.get_name()),
                              value_or_default(samples.get_persistence_mode()),
                              value_or_default(samples.get_num_cores()),
                              fmt::join(value_or_default(samples.get_performance_state()), ", "),
                              fmt::join(value_or_default(samples.get_utilization_gpu()), ", "),
                              fmt::join(value_or_default(samples.get_utilization_mem()), ", "));
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string nvml_clock_samples::generate_yaml_string() const {
    std::string str{ "    clock:\n" };

    // adaptive clock status
    if (this->adaptive_clock_status_.has_value()) {
        str += fmt::format("      adaptive_clock_status:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->adaptive_clock_status_.value() == NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED);
    }
    // maximum SM clock
    if (this->clock_sm_max_.has_value()) {
        str += fmt::format("      clock_sm_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_sm_max_.value());
    }
    // minimum memory clock
    if (this->clock_mem_min_.has_value()) {
        str += fmt::format("      clock_mem_min:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_mem_min_.value());
    }
    // maximum memory clock
    if (this->clock_mem_max_.has_value()) {
        str += fmt::format("      clock_mem_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_mem_max_.value());
    }
    // minimum graph clock
    if (this->clock_graph_min_.has_value()) {
        str += fmt::format("      clock_gpu_min:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_graph_min_.value());
    }
    // maximum graph clock
    if (this->clock_graph_max_.has_value()) {
        str += fmt::format("      clock_gpu_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_graph_max_.value());
    }

    // SM clock
    if (this->clock_sm_.has_value()) {
        str += fmt::format("      clock_sm:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_sm_.value(), ", "));
    }
    // memory clock
    if (this->clock_mem_.has_value()) {
        str += fmt::format("      clock_mem:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_mem_.value(), ", "));
    }
    // graph clock
    if (this->clock_graph_.has_value()) {
        str += fmt::format("      clock_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_graph_.value(), ", "));
    }
    // clock throttle reason
    if (this->clock_throttle_reason_.has_value()) {
        str += fmt::format("      clock_throttle_reason:\n"
                           "        unit: \"bitmask\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_throttle_reason_.value(), ", "));
    }
    // clock is auto-boosted
    if (this->auto_boosted_clocks_.has_value()) {
        str += fmt::format("      auto_boosted_clocks:\n"
                           "        unit: \"bool\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->auto_boosted_clocks_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_clock_samples &samples) {
    return out << fmt::format("adaptive_clock_status: {}\n"
                              "clock_graph_min: {}\n"
                              "clock_graph_max: {}\n"
                              "clock_sm_max: {}\n"
                              "clock_mem_min: {}\n"
                              "clock_mem_max: {}\n"
                              "clock_graph: [{}]\n"
                              "clock_sm: [{}]\n"
                              "clock_mem: [{}]\n"
                              "clock_throttle_reason: [{}]\n"
                              "auto_boosted_clocks: [{}]",
                              value_or_default(samples.get_adaptive_clock_status()),
                              value_or_default(samples.get_clock_graph_min()),
                              value_or_default(samples.get_clock_graph_max()),
                              value_or_default(samples.get_clock_sm_max()),
                              value_or_default(samples.get_clock_mem_min()),
                              value_or_default(samples.get_clock_mem_max()),
                              fmt::join(value_or_default(samples.get_clock_graph()), ", "),
                              fmt::join(value_or_default(samples.get_clock_sm()), ", "),
                              fmt::join(value_or_default(samples.get_clock_mem()), ", "),
                              fmt::join(value_or_default(samples.get_clock_throttle_reason()), ", "),
                              fmt::join(value_or_default(samples.get_auto_boosted_clocks()), ", "));
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string nvml_power_samples::generate_yaml_string() const {
    std::string str{ "    power:\n" };

    // the power management mode
    if (this->power_management_mode_.has_value()) {
        str += fmt::format("      power_management_mode:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->power_management_mode_.value());
    }
    // power management limit
    if (this->power_management_limit_.has_value()) {
        str += fmt::format("      power_management_limit:\n"
                           "        unit: \"mW\"\n"
                           "        values: {}\n",
                           this->power_management_limit_.value());
    }
    // power enforced limit
    if (this->power_enforced_limit_.has_value()) {
        str += fmt::format("      power_enforced_limit:\n"
                           "        unit: \"mW\"\n"
                           "        values: {}\n",
                           this->power_enforced_limit_.value());
    }

    // power state
    if (this->power_state_.has_value()) {
        str += fmt::format("      power_state:\n"
                           "        unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->power_state_.value(), ", "));
    }
    // current power usage
    if (this->power_usage_.has_value()) {
        str += fmt::format("      power_usage:\n"
                           "        unit: \"mW\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->power_usage_.value(), ", "));
    }
    // total energy consumed
    if (this->power_total_energy_consumption_.has_value()) {
        decltype(nvml_power_samples::power_total_energy_consumption_)::value_type consumed_energy(this->power_total_energy_consumption_->size());
#pragma omp parallel for
        for (std::size_t i = 0; i < consumed_energy.size(); ++i) {
            consumed_energy[i] = this->power_total_energy_consumption_.value()[i] - this->power_total_energy_consumption_->front();
        }
        str += fmt::format("      power_total_energy_consumed:\n"
                           "        unit: \"J\"\n"
                           "        values: [{}]\n",
                           fmt::join(consumed_energy, ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples) {
    return out << fmt::format("power_management_mode: {}\n"
                              "power_management_limit: {}\n"
                              "power_enforced_limit: {}\n"
                              "power_state: [{}]\n"
                              "power_usage: [{}]\n"
                              "power_total_energy_consumption: [{}]",
                              value_or_default(samples.get_power_management_mode()),
                              value_or_default(samples.get_power_management_limit()),
                              value_or_default(samples.get_power_enforced_limit()),
                              fmt::join(value_or_default(samples.get_power_state()), ", "),
                              fmt::join(value_or_default(samples.get_power_usage()), ", "),
                              fmt::join(value_or_default(samples.get_power_total_energy_consumption()), ", "));
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string nvml_memory_samples::generate_yaml_string() const {
    std::string str{ "    memory:\n" };

    // total memory size
    if (this->memory_total_.has_value()) {
        str += fmt::format("      memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->memory_total_.value());
    }
    // maximum PCIe link speed
    if (this->pcie_link_max_speed_.has_value()) {
        str += fmt::format("      pcie_max_bandwidth:\n"
                           "        unit: \"MBPS\"\n"
                           "        values: {}\n",
                           this->pcie_link_max_speed_.value());
    }
    // memory bus width
    if (this->memory_bus_width_.has_value()) {
        str += fmt::format("      memory_bus_width:\n"
                           "        unit: \"Bit\"\n"
                           "        values: {}\n",
                           this->memory_bus_width_.value());
    }
    // maximum PCIe link generation
    if (this->max_pcie_link_generation_.has_value()) {
        str += fmt::format("      max_pcie_link_generation:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->max_pcie_link_generation_.value());
    }

    // free memory size
    if (this->memory_free_.has_value()) {
        str += fmt::format("      memory_free:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_free_.value(), ", "));
    }
    // used memory size
    if (this->memory_used_.has_value()) {
        str += fmt::format("      memory_used:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_used_.value(), ", "));
    }
    // PCIe link speed
    if (this->pcie_link_speed_.has_value()) {
        str += fmt::format("      pcie_bandwidth:\n"
                           "        unit: \"MBPS\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->pcie_link_speed_.value(), ", "));
    }
    // PCIe link width
    if (this->pcie_link_width_.has_value()) {
        str += fmt::format("      pcie_link_width:\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->pcie_link_width_.value(), ", "));
    }
    // PCIe link generation
    if (this->pcie_link_generation_.has_value()) {
        str += fmt::format("      pcie_link_generation:\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->pcie_link_generation_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_memory_samples &samples) {
    return out << fmt::format("memory_total: {}\n"
                              "pcie_link_max_speed: {}\n"
                              "memory_bus_width: {}\n"
                              "max_pcie_link_generation: {}\n"
                              "memory_free: [{}]\n"
                              "memory_used: [{}]\n"
                              "pcie_link_speed: [{}]\n"
                              "pcie_link_width: [{}]\n"
                              "pcie_link_generation: [{}]",
                              value_or_default(samples.get_memory_total()),
                              value_or_default(samples.get_pcie_link_max_speed()),
                              value_or_default(samples.get_memory_bus_width()),
                              value_or_default(samples.get_max_pcie_link_generation()),
                              fmt::join(value_or_default(samples.get_memory_free()), ", "),
                              fmt::join(value_or_default(samples.get_memory_used()), ", "),
                              fmt::join(value_or_default(samples.get_pcie_link_speed()), ", "),
                              fmt::join(value_or_default(samples.get_pcie_link_width()), ", "),
                              fmt::join(value_or_default(samples.get_pcie_link_generation()), ", "));
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string nvml_temperature_samples::generate_yaml_string() const {
    std::string str{ "    temperature:\n" };

    // number of fans
    if (this->num_fans_.has_value()) {
        str += fmt::format("      num_fans:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_fans_.value());
    }
    // min fan speed
    if (this->min_fan_speed_.has_value()) {
        str += fmt::format("      min_fan_speed:\n"
                           "        unit: \"percentage\"\n"
                           "        values: {}\n",
                           this->min_fan_speed_.value());
    }
    // max fan speed
    if (this->max_fan_speed_.has_value()) {
        str += fmt::format("      max_fan_speed:\n"
                           "        unit: \"percentage\"\n"
                           "        values: {}\n",
                           this->max_fan_speed_.value());
    }
    // temperature threshold GPU max
    if (this->temperature_threshold_gpu_max_.has_value()) {
        str += fmt::format("      temperature_gpu_max:\n"
                           "        unit: \"°C\"\n"
                           "        values: {}\n",
                           this->temperature_threshold_gpu_max_.value());
    }
    // temperature threshold memory max
    if (this->temperature_threshold_mem_max_.has_value()) {
        str += fmt::format("      temperature_mem_max:\n"
                           "        unit: \"°C\"\n"
                           "        values: {}\n",
                           this->temperature_threshold_mem_max_.value());
    }

    // fan speed
    if (this->fan_speed_.has_value()) {
        str += fmt::format("      fan_speed:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->fan_speed_.value(), ", "));
    }
    // temperature GPU
    if (this->temperature_gpu_.has_value()) {
        str += fmt::format("      temperature_gpu:\n"
                           "        unit: \"°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_gpu_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples &samples) {
    return out << fmt::format("num_fans: {}\n"
                              "min_fan_speed: {}\n"
                              "max_fan_speed: {}\n"
                              "temperature_threshold_gpu_max: {}\n"
                              "temperature_threshold_mem_max: {}\n"
                              "fan_speed: [{}]\n"
                              "temperature_gpu: [{}]",
                              value_or_default(samples.get_num_fans()),
                              value_or_default(samples.get_min_fan_speed()),
                              value_or_default(samples.get_max_fan_speed()),
                              value_or_default(samples.get_temperature_threshold_gpu_max()),
                              value_or_default(samples.get_temperature_threshold_mem_max()),
                              fmt::join(value_or_default(samples.get_fan_speed()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_gpu()), ", "));
}

}  // namespace plssvm::detail::tracking
