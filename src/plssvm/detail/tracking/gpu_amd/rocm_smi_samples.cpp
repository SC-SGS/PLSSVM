/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_amd/rocm_smi_samples.hpp"

#include "plssvm/detail/operators.hpp"         // operators namespace
#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::value_or_default

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "rocm_smi/rocm_smi.h"  // RSMI_MAX_FAN_SPEED

#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <ostream>   // std::ostream
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string rocm_smi_general_samples::generate_yaml_string() const {
    std::string str{ "    general:\n" };

    // device name
    if (this->name_.has_value()) {
        str += fmt::format("      name:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->name_.value());
    }

    // performance state
    if (this->performance_level_.has_value()) {
        str += fmt::format("      performance_state:\n"
                           "        unit: \"int - see rsmi_dev_perf_level_t\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->performance_level_.value(), ", "));
    }
    // device compute utilization
    if (this->utilization_gpu_.has_value()) {
        str += fmt::format("      utilization_gpu:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->utilization_gpu_.value(), ", "));
    }
    // device memory utilization
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

std::ostream &operator<<(std::ostream &out, const rocm_smi_general_samples &samples) {
    return out << fmt::format("name: {}\n"
                              "performance_level: [{}]\n"
                              "utilization_gpu: [{}]\n"
                              "utilization_mem: [{}]",
                              value_or_default(samples.get_name()),
                              fmt::join(value_or_default(samples.get_performance_level()), ", "),
                              fmt::join(value_or_default(samples.get_utilization_gpu()), ", "),
                              fmt::join(value_or_default(samples.get_utilization_mem()), ", "));
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string rocm_smi_clock_samples::generate_yaml_string() const {
    std::string str{ "    clock:\n" };

    // socket clock min frequencies
    if (this->clock_socket_min_.has_value()) {
        str += fmt::format("      clock_socket_min:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_socket_min_.value());
    }
    // socket clock max frequencies
    if (this->clock_socket_max_.has_value()) {
        str += fmt::format("      clock_socket_max:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_socket_max_.value());
    }

    // memory clock min frequencies
    if (this->clock_memory_min_.has_value()) {
        str += fmt::format("      clock_memory_min:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_memory_min_.value());
    }
    // memory clock max frequencies
    if (this->clock_memory_max_.has_value()) {
        str += fmt::format("      clock_memory_max:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_memory_max_.value());
    }

    // system clock min frequencies
    if (this->clock_system_min_.has_value()) {
        str += fmt::format("      clock_gpu_min:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_system_min_.value());
    }
    // system clock max frequencies
    if (this->clock_system_max_.has_value()) {
        str += fmt::format("      clock_gpu_max:\n"
                           "        unit: \"Hz\"\n"
                           "        values: {}\n",
                           this->clock_system_max_.value());
    }

    // socket clock frequency
    if (this->clock_socket_.has_value()) {
        str += fmt::format("      clock_socket:\n"
                           "        unit: \"Hz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_socket_.value(), ", "));
    }
    // memory clock frequency
    if (this->clock_memory_.has_value()) {
        str += fmt::format("      clock_memory:\n"
                           "        unit: \"Hz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_memory_.value(), ", "));
    }
    // system clock frequency
    if (this->clock_system_.has_value()) {
        str += fmt::format("      clock_gpu:\n"
                           "        unit: \"Hz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_system_.value(), ", "));
    }
    // overdrive level
    if (this->overdrive_level_.has_value()) {
        str += fmt::format("      overdrive_level:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->overdrive_level_.value(), ", "));
    }
    // memory overdrive level
    if (this->memory_overdrive_level_.has_value()) {
        str += fmt::format("      memory_overdrive_level:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_overdrive_level_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_clock_samples &samples) {
    return out << fmt::format("clock_system_min: {}\n"
                              "clock_system_max: {}\n"
                              "clock_socket_min: {}\n"
                              "clock_socket_max: {}\n"
                              "clock_memory_min: {}\n"
                              "clock_memory_max: {}\n"
                              "clock_system: [{}]\n"
                              "clock_socket: [{}]\n"
                              "clock_memory: [{}]\n"
                              "overdrive_level: [{}]\n"
                              "memory_overdrive_level: [{}]",
                              value_or_default(samples.get_clock_system_min()),
                              value_or_default(samples.get_clock_system_max()),
                              value_or_default(samples.get_clock_socket_min()),
                              value_or_default(samples.get_clock_socket_max()),
                              value_or_default(samples.get_clock_memory_min()),
                              value_or_default(samples.get_clock_memory_max()),
                              fmt::join(value_or_default(samples.get_clock_system()), ", "),
                              fmt::join(value_or_default(samples.get_clock_socket()), ", "),
                              fmt::join(value_or_default(samples.get_clock_memory()), ", "),
                              fmt::join(value_or_default(samples.get_overdrive_level()), ", "),
                              fmt::join(value_or_default(samples.get_memory_overdrive_level()), ", "));
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string rocm_smi_power_samples::generate_yaml_string() const {
    std::string str{ "    power:\n" };

    // default power cap
    if (this->power_default_cap_.has_value()) {
        str += fmt::format("      power_management_limit:\n"
                           "        unit: \"muW\"\n"
                           "        values: {}\n",
                           this->power_default_cap_.value());
    }
    // power cap
    if (this->power_cap_.has_value()) {
        str += fmt::format("      power_enforced_limit:\n"
                           "        unit: \"muW\"\n"
                           "        values: {}\n",
                           this->power_cap_.value());
    }
    // power measurement type
    if (this->power_type_.has_value()) {
        str += fmt::format("      power_measurement_type:\n"
                           "        unit: \"string\"\n"
                           "        values: {}\n",
                           this->power_type_.value());
    }
    // available power levels
    if (this->available_power_profiles_.has_value()) {
        str += fmt::format("      available_power_profiles:\n"
                           "        unit: \"string\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->available_power_profiles_.value(), ", "));
    }

    // current power usage
    if (this->power_usage_.has_value()) {
        str += fmt::format("      power_usage:\n"
                           "        unit: \"muW\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->power_usage_.value(), ", "));
    }
    // total energy consumed
    if (this->power_total_energy_consumption_.has_value()) {
        decltype(rocm_smi_power_samples::power_total_energy_consumption_)::value_type consumed_energy(this->power_total_energy_consumption_->size());
#pragma omp parallel for
        for (std::size_t i = 0; i < consumed_energy.size(); ++i) {
            consumed_energy[i] = this->power_total_energy_consumption_.value()[i] - this->power_total_energy_consumption_->front();
        }
        str += fmt::format("      power_total_energy_consumed:\n"
                           "        unit: \"muJ\"\n"
                           "        values: [{}]\n",
                           fmt::join(consumed_energy, ", "));
    }
    // current power level
    if (this->power_profile_.has_value()) {
        str += fmt::format("      power_profile:\n"
                           "        unit: \"string\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->power_profile_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_power_samples &samples) {
    return out << fmt::format("power_default_cap: {}\n"
                              "power_cap: {}\n"
                              "power_type: {}\n"
                              "available_power_profiles: [{}]\n"
                              "power_usage: [{}]\n"
                              "power_total_energy_consumption: [{}]\n"
                              "power_profile: [{}]",
                              value_or_default(samples.get_power_default_cap()),
                              value_or_default(samples.get_power_cap()),
                              value_or_default(samples.get_power_type()),
                              fmt::join(value_or_default(samples.get_available_power_profiles()), ", "),
                              fmt::join(value_or_default(samples.get_power_usage()), ", "),
                              fmt::join(value_or_default(samples.get_power_total_energy_consumption()), ", "),
                              fmt::join(value_or_default(samples.get_power_profile()), ", "));
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string rocm_smi_memory_samples::generate_yaml_string() const {
    std::string str{ "    memory:\n" };

    // total memory
    if (this->memory_total_.has_value()) {
        str += fmt::format("      memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->memory_total_.value());
    }
    // total visible memory
    if (this->visible_memory_total_.has_value()) {
        str += fmt::format("      visible_memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->visible_memory_total_.value());
    }
    // min number of PCIe lanes
    if (this->min_num_pcie_lanes_.has_value()) {
        str += fmt::format("      min_num_pcie_lanes:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->min_num_pcie_lanes_.value());
    }
    // max number of PCIe lanes
    if (this->max_num_pcie_lanes_.has_value()) {
        str += fmt::format("      max_num_pcie_lanes:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->max_num_pcie_lanes_.value());
    }

    // used memory
    if (this->memory_used_.has_value()) {
        str += fmt::format("      memory_used:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_used_.value(), ", "));
    }
    // free memory
    if (this->memory_used_.has_value() && this->memory_total_.has_value()) {
        using namespace plssvm::operators;
        decltype(rocm_smi_memory_samples::memory_used_)::value_type memory_free(this->memory_used_->size(), this->memory_total_.value());
        memory_free -= this->memory_used_.value();
        str += fmt::format("      memory_free:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(memory_free, ", "));
    }

    // PCIe bandwidth
    if (this->pcie_transfer_rate_.has_value()) {
        str += fmt::format("      pcie_bandwidth:\n"
                           "        unit: \"T/s\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->pcie_transfer_rate_.value(), ", "));
    }
    // number of PCIe lanes
    if (this->num_pcie_lanes_.has_value()) {
        str += fmt::format("      pcie_num_lanes:\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->num_pcie_lanes_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_memory_samples &samples) {
    return out << fmt::format("memory_total: {}\n"
                              "visible_memory_total: {}\n"
                              "min_num_pcie_lanes: {}\n"
                              "max_num_pcie_lanes: {}\n"
                              "memory_used: [{}]\n"
                              "pcie_transfer_rate: [{}]\n"
                              "num_pcie_lanes: [{}]",
                              value_or_default(samples.get_memory_total()),
                              value_or_default(samples.get_visible_memory_total()),
                              value_or_default(samples.get_min_num_pcie_lanes()),
                              value_or_default(samples.get_max_num_pcie_lanes()),
                              fmt::join(value_or_default(samples.get_memory_used()), ", "),
                              fmt::join(value_or_default(samples.get_pcie_transfer_rate()), ", "),
                              fmt::join(value_or_default(samples.get_num_pcie_lanes()), ", "));
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string rocm_smi_temperature_samples::generate_yaml_string() const {
    std::string str{ "    temperature:\n" };

    // number of fans (emulated)
    if (this->num_fans_.has_value()) {
        str += fmt::format("      num_fans:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_fans_.value());
    }
    // maximum fan speed
    if (this->max_fan_speed_.has_value()) {
        str += fmt::format("      max_fan_speed:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->max_fan_speed_.value());
    }
    // minimum GPU edge temperature
    if (this->temperature_edge_min_.has_value()) {
        str += fmt::format("      temperature_gpu_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_edge_min_.value());
    }
    // maximum GPU edge temperature
    if (this->temperature_edge_max_.has_value()) {
        str += fmt::format("      temperature_gpu_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_edge_max_.value());
    }
    // minimum GPU hotspot temperature
    if (this->temperature_hotspot_min_.has_value()) {
        str += fmt::format("      temperature_hotspot_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hotspot_min_.value());
    }
    // maximum GPU hotspot temperature
    if (this->temperature_hotspot_max_.has_value()) {
        str += fmt::format("      temperature_hotspot_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hotspot_max_.value());
    }
    // minimum GPU memory temperature
    if (this->temperature_memory_min_.has_value()) {
        str += fmt::format("      temperature_memory_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_memory_min_.value());
    }
    // maximum GPU memory temperature
    if (this->temperature_memory_max_.has_value()) {
        str += fmt::format("      temperature_memory_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_memory_max_.value());
    }
    // minimum GPU HBM 0 temperature
    if (this->temperature_hbm_0_min_.has_value()) {
        str += fmt::format("      temperature_hbm_0_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_0_min_.value());
    }
    // maximum GPU HBM 0 temperature
    if (this->temperature_hbm_0_max_.has_value()) {
        str += fmt::format("      temperature_hbm_0_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_0_max_.value());
    }
    // minimum GPU HBM 1 temperature
    if (this->temperature_hbm_1_min_.has_value()) {
        str += fmt::format("      temperature_hbm_1_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_1_min_.value());
    }
    // maximum GPU HBM 1 temperature
    if (this->temperature_hbm_1_max_.has_value()) {
        str += fmt::format("      temperature_hbm_1_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_1_max_.value());
    }
    // minimum GPU HBM 2 temperature
    if (this->temperature_hbm_2_min_.has_value()) {
        str += fmt::format("      temperature_hbm_2_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_2_min_.value());
    }
    // maximum GPU HBM 2 temperature
    if (this->temperature_hbm_2_max_.has_value()) {
        str += fmt::format("      temperature_hbm_2_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_2_max_.value());
    }
    // minimum GPU HBM 3 temperature
    if (this->temperature_hbm_3_min_.has_value()) {
        str += fmt::format("      temperature_hbm_3_min:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_3_min_.value());
    }
    // maximum GPU HBM 3 temperature
    if (this->temperature_hbm_3_max_.has_value()) {
        str += fmt::format("      temperature_hbm_3_max:\n"
                           "        unit: \"m°C\"\n"
                           "        values: {}\n",
                           this->temperature_hbm_3_max_.value());
    }

    // fan speed
    if (this->fan_speed_.has_value()) {
        std::vector<double> fan_speed_percent(this->fan_speed_->size());
#pragma omp parallel for
        for (std::size_t i = 0; i < fan_speed_percent.size(); ++i) {
            fan_speed_percent[i] = static_cast<double>(this->fan_speed_.value()[i]) / static_cast<double>(RSMI_MAX_FAN_SPEED);
        }
        str += fmt::format("      fan_speed:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(fan_speed_percent, ", "));
    }
    // GPU edge temperature
    if (this->temperature_edge_.has_value()) {
        str += fmt::format("      temperature_gpu:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_edge_.value(), ", "));
    }
    // GPU hotspot temperature
    if (this->temperature_hotspot_.has_value()) {
        str += fmt::format("      temperature_hotspot:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_hotspot_.value(), ", "));
    }
    // GPU memory temperature
    if (this->temperature_memory_.has_value()) {
        str += fmt::format("      temperature_memory:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_memory_.value(), ", "));
    }
    // GPU HBM 0 temperature
    if (this->temperature_hbm_0_.has_value()) {
        str += fmt::format("      temperature_hbm_0:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_hbm_0_.value(), ", "));
    }
    // GPU HBM 1 temperature
    if (this->temperature_hbm_1_.has_value()) {
        str += fmt::format("      temperature_hbm_1:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_hbm_1_.value(), ", "));
    }
    // GPU HBM 2 temperature
    if (this->temperature_hbm_2_.has_value()) {
        str += fmt::format("      temperature_hbm_2:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_hbm_2_.value(), ", "));
    }
    // GPU HBM 3 temperature
    if (this->temperature_hbm_3_.has_value()) {
        str += fmt::format("      temperature_hbm_3:\n"
                           "        unit: \"m°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_hbm_3_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_temperature_samples &samples) {
    return out << fmt::format("num_fans: {}\n"
                              "max_fan_speed: {}\n"
                              "temperature_edge_min: {}\n"
                              "temperature_edge_max: {}\n"
                              "temperature_hotspot_min: {}\n"
                              "temperature_hotspot_max: {}\n"
                              "temperature_memory_min: {}\n"
                              "temperature_memory_max: {}\n"
                              "temperature_hbm_0_min: {}\n"
                              "temperature_hbm_0_max: {}\n"
                              "temperature_hbm_1_min: {}\n"
                              "temperature_hbm_1_max: {}\n"
                              "temperature_hbm_2_min: {}\n"
                              "temperature_hbm_2_max: {}\n"
                              "temperature_hbm_3_min: {}\n"
                              "temperature_hbm_3_max: {}\n"
                              "fan_speed: [{}]\n"
                              "temperature_edge: [{}]\n"
                              "temperature_hotspot: [{}]\n"
                              "temperature_memory: [{}]\n"
                              "temperature_hbm_0: [{}]\n"
                              "temperature_hbm_1: [{}]\n"
                              "temperature_hbm_2: [{}]\n"
                              "temperature_hbm_3: [{}]",
                              value_or_default(samples.get_num_fans()),
                              value_or_default(samples.get_max_fan_speed()),
                              value_or_default(samples.get_temperature_edge_min()),
                              value_or_default(samples.get_temperature_edge_max()),
                              value_or_default(samples.get_temperature_hotspot_min()),
                              value_or_default(samples.get_temperature_hotspot_max()),
                              value_or_default(samples.get_temperature_memory_min()),
                              value_or_default(samples.get_temperature_memory_max()),
                              value_or_default(samples.get_temperature_hbm_0_min()),
                              value_or_default(samples.get_temperature_hbm_0_max()),
                              value_or_default(samples.get_temperature_hbm_1_min()),
                              value_or_default(samples.get_temperature_hbm_1_max()),
                              value_or_default(samples.get_temperature_hbm_2_min()),
                              value_or_default(samples.get_temperature_hbm_2_max()),
                              value_or_default(samples.get_temperature_hbm_3_min()),
                              value_or_default(samples.get_temperature_hbm_3_max()),
                              fmt::join(value_or_default(samples.get_fan_speed()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_edge()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_hotspot()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_memory()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_hbm_0()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_hbm_1()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_hbm_2()), ", "),
                              fmt::join(value_or_default(samples.get_temperature_hbm_3()), ", "));
}

}  // namespace plssvm::detail::tracking
