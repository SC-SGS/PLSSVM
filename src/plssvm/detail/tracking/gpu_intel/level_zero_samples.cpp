/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_intel/level_zero_samples.hpp"

#include "plssvm/detail/operators.hpp"         // operators namespace
#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::value_or_default
#include "plssvm/detail/type_traits.hpp"       // plssvm::detail::{remove_cvref_t, is_vector}

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <cstddef>      // std::size_t
#include <optional>     // std::optional
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail::tracking {

template <typename MapType>
void append_map_values(std::string &str, const std::string_view entry_name, const MapType &map) {
    if (map.has_value()) {
        for (const auto &[key, value] : map.value()) {
            if constexpr (detail::is_vector_v<detail::remove_cvref_t<decltype(value)>>) {
                str += fmt::format("{}_{}: [{}]\n", entry_name, key, fmt::join(value, ", "));
            } else {
                str += fmt::format("{}_{}: {}\n", entry_name, key, value);
            }
        }
    }
}

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string level_zero_general_samples::generate_yaml_string() const {
    std::string str{ "    general:\n" };

    // the model name
    if (this->name_.has_value()) {
        str += fmt::format("      model_name:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->name_.value());
    }
    // the standby mode
    if (this->standby_mode_.has_value()) {
        str += fmt::format("      standby_mode:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->standby_mode_.value());
    }
    // the number of threads per EU unit
    if (this->num_threads_per_eu_.has_value()) {
        str += fmt::format("      num_threads_per_eu:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_threads_per_eu_.value());
    }
    // the EU SIMD width
    if (this->eu_simd_width_.has_value()) {
        str += fmt::format("      physical_eu_simd_width:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->eu_simd_width_.value());
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_general_samples &samples) {
    return out << fmt::format("name: {}\n"
                              "standby_mode: {}\n"
                              "num_threads_per_eu: {}\n"
                              "eu_simd_width: {}",
                              value_or_default(samples.get_name()),
                              value_or_default(samples.get_standby_mode()),
                              value_or_default(samples.get_num_threads_per_eu()),
                              value_or_default(samples.get_eu_simd_width()));
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string level_zero_clock_samples::generate_yaml_string() const {
    std::string str{ "    clock:\n" };

    // minimum GPU core clock
    if (this->clock_gpu_min_.has_value()) {
        str += fmt::format("      clock_gpu_min:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_gpu_min_.value());
    }
    // maximum GPU core clock
    if (this->clock_gpu_max_.has_value()) {
        str += fmt::format("      clock_gpu_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_gpu_max_.value());
    }
    // all possible GPU core clock frequencies
    if (this->available_clocks_gpu_.has_value()) {
        str += fmt::format("      available_clocks_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->available_clocks_gpu_.value(), ", "));
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
    // all possible memory clock frequencies
    if (this->available_clocks_mem_.has_value()) {
        str += fmt::format("      available_clocks_mem:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->available_clocks_mem_.value(), ", "));
    }

    // the maximum GPU core frequency based on the current TDP limit
    if (this->tdp_frequency_limit_gpu_.has_value()) {
        str += fmt::format("      tdp_frequency_limit_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->tdp_frequency_limit_gpu_.value(), ", "));
    }
    // the current GPU core clock frequency
    if (this->clock_gpu_.has_value()) {
        str += fmt::format("      clock_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_gpu_.value(), ", "));
    }
    // the current GPU core throttle reason
    if (this->throttle_reason_gpu_.has_value()) {
        str += fmt::format("      throttle_reason_gpu:\n"
                           "        unit: \"bitmask\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->throttle_reason_gpu_.value(), ", "));
    }
    // the maximum memory frequency based on the current TDP limit
    if (this->tdp_frequency_limit_mem_.has_value()) {
        str += fmt::format("      tdp_frequency_limit_mem:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->tdp_frequency_limit_mem_.value(), ", "));
    }
    // the current memory clock frequency
    if (this->clock_mem_.has_value()) {
        str += fmt::format("      clock_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->clock_mem_.value(), ", "));
    }
    // the current memory throttle reason
    if (this->throttle_reason_mem_.has_value()) {
        str += fmt::format("      throttle_reason_gpu:\n"
                           "        unit: \"bitmask\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->throttle_reason_mem_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_clock_samples &samples) {
    return out << fmt::format("clock_gpu_min: {}\n"
                              "clock_gpu_max: {}\n"
                              "available_clocks_gpu: [{}]\n"
                              "clock_mem_min: {}\n"
                              "clock_mem_max: {}\n"
                              "available_clocks_mem: [{}]\n"
                              "tdp_frequency_limit_gpu: [{}]\n"
                              "clock_gpu: [{}]\n"
                              "throttle_reason_gpu: [{}]\n"
                              "tdp_frequency_limit_mem: [{}]\n"
                              "clock_mem: [{}]\n"
                              "throttle_reason_mem: [{}]",
                              value_or_default(samples.get_clock_gpu_min()),
                              value_or_default(samples.get_clock_gpu_max()),
                              fmt::join(value_or_default(samples.get_available_clocks_gpu()), ", "),
                              value_or_default(samples.get_clock_mem_min()),
                              value_or_default(samples.get_clock_mem_max()),
                              fmt::join(value_or_default(samples.get_available_clocks_mem()), ", "),
                              fmt::join(value_or_default(samples.get_tdp_frequency_limit_gpu()), ", "),
                              fmt::join(value_or_default(samples.get_clock_gpu()), ", "),
                              fmt::join(value_or_default(samples.get_throttle_reason_gpu()), ", "),
                              fmt::join(value_or_default(samples.get_tdp_frequency_limit_mem()), ", "),
                              fmt::join(value_or_default(samples.get_clock_mem()), ", "),
                              fmt::join(value_or_default(samples.get_throttle_reason_mem()), ", "));
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string level_zero_power_samples::generate_yaml_string() const {
    std::string str{ "    power:\n" };

    // flag whether the energy threshold is enabled
    if (this->energy_threshold_enabled_.has_value()) {
        str += fmt::format("      energy_threshold_enabled:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->energy_threshold_enabled_.value());
    }
    // the energy threshold
    if (this->energy_threshold_.has_value()) {
        str += fmt::format("      energy_threshold:\n"
                           "        unit: \"J\"\n"
                           "        values: {}\n",
                           this->energy_threshold_.value());
    }

    // the total consumed energy
    if (this->power_total_energy_consumption_.has_value()) {
        decltype(level_zero_power_samples::power_total_energy_consumption_)::value_type consumed_energy(this->power_total_energy_consumption_->size());
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

std::ostream &operator<<(std::ostream &out, const level_zero_power_samples &samples) {
    return out << fmt::format("energy_threshold_enabled: {}\n"
                              "energy_threshold: {}\n"
                              "power_total_energy_consumption: [{}]",
                              value_or_default(samples.get_energy_threshold_enabled()),
                              value_or_default(samples.get_energy_threshold()),
                              fmt::join(value_or_default(samples.get_power_total_energy_consumption()), ", "));
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string level_zero_memory_samples::generate_yaml_string() const {
    std::string str{ "    memory:\n" };

    // the total memory
    if (this->memory_total_.has_value()) {
        for (const auto &[key, value] : this->memory_total_.value()) {
            str += fmt::format("      memory_total_{}:\n"
                               "        unit: \"B\"\n"
                               "        values: {}\n",
                               key,
                               value);
        }
    }
    // the total allocatable memory
    if (this->allocatable_memory_total_.has_value()) {
        for (const auto &[key, value] : this->allocatable_memory_total_.value()) {
            str += fmt::format("      allocatable_memory_total_{}:\n"
                               "        unit: \"B\"\n"
                               "        values: {}\n",
                               key,
                               value);
        }
    }
    // the pcie max bandwidth
    if (this->pcie_link_max_speed_.has_value()) {
        str += fmt::format("      pcie_max_bandwidth:\n"
                           "        unit: \"BPS\"\n"
                           "        values: {}\n",
                           this->pcie_link_max_speed_.value());
    }
    // the pcie link width
    if (this->pcie_max_width_.has_value()) {
        str += fmt::format("      max_pcie_link_width:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->pcie_max_width_.value());
    }
    // the pcie generation
    if (this->max_pcie_link_generation_.has_value()) {
        str += fmt::format("      max_pcie_link_generation:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->max_pcie_link_generation_.value());
    }
    // the memory bus width
    if (this->bus_width_.has_value()) {
        for (const auto &[key, value] : this->bus_width_.value()) {
            str += fmt::format("      memory_bus_width_{}:\n"
                               "        unit: \"Bit\"\n"
                               "        values: {}\n",
                               key,
                               value);
        }
    }
    // the number of memory channels
    if (this->num_channels_.has_value()) {
        for (const auto &[key, value] : this->num_channels_.value()) {
            str += fmt::format("      memory_num_channels_{}:\n"
                               "        unit: \"int\"\n"
                               "        values: {}\n",
                               key,
                               value);
        }
    }
    // the memory location (system or device)
    if (this->location_.has_value()) {
        for (const auto &[key, value] : this->location_.value()) {
            str += fmt::format("      memory_location_{}:\n"
                               "        unit: \"string\"\n"
                               "        values: \"{}\"\n",
                               key,
                               value);
        }
    }

    // the currently free and used memory
    if (this->memory_free_.has_value()) {
        for (const auto &[key, value] : this->memory_free_.value()) {
            str += fmt::format("      memory_free_{}:\n"
                               "        unit: \"string\"\n"
                               "        values: [{}]\n",
                               key,
                               fmt::join(value, ", "));

            // calculate the used memory
            if (this->allocatable_memory_total_.has_value()) {
                using namespace plssvm::operators;
                decltype(level_zero_memory_samples::memory_free_)::value_type::mapped_type memory_used(value.size(), this->allocatable_memory_total_->at(key));
                memory_used -= value;
                str += fmt::format("      memory_used_{}:\n"
                                   "        unit: \"string\"\n"
                                   "        values: [{}]\n",
                                   key,
                                   fmt::join(memory_used, ", "));
            }
        }
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

std::ostream &operator<<(std::ostream &out, const level_zero_memory_samples &samples) {
    std::string str{};

    append_map_values(str, "memory_total", samples.get_memory_total());
    append_map_values(str, "allocatable_memory_total", samples.get_allocatable_memory_total());

    str += fmt::format("pcie_link_max_speed: {}\n"
                       "pcie_max_width: {}\n"
                       "max_pcie_link_generation: {}\n",
                       value_or_default(samples.get_pcie_link_max_speed()),
                       value_or_default(samples.get_pcie_max_width()),
                       value_or_default(samples.get_max_pcie_link_generation()));

    append_map_values(str, "bus_width", samples.get_bus_width());
    append_map_values(str, "num_channels", samples.get_num_channels());
    append_map_values(str, "location", samples.get_location());
    append_map_values(str, "memory_free", samples.get_memory_free());

    str += fmt::format("pcie_link_speed: [{}]\n"
                       "pcie_link_width: [{}]\n"
                       "pcie_link_generation: [{}]",
                       fmt::join(value_or_default(samples.get_pcie_link_speed()), ", "),
                       fmt::join(value_or_default(samples.get_pcie_link_width()), ", "),
                       fmt::join(value_or_default(samples.get_pcie_link_generation()), ", "));

    return out << str;
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string level_zero_temperature_samples::generate_yaml_string() const {
    std::string str{ "    temperature:\n" };

    // the maximum sensor temperature
    if (this->temperature_max_.has_value()) {
        for (const auto &[key, value] : this->temperature_max_.value()) {
            str += fmt::format("      temperature_{}_max:\n"
                               "        unit: \"°C\"\n"
                               "        values: {}\n",
                               key,
                               value);
        }
    }

    // the current PSU temperatures
    if (this->temperature_psu_.has_value()) {
        str += fmt::format("      temperature_psu:\n"
                           "        unit: \"°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->temperature_psu_.value(), ", "));
    }
    // the current sensor temperatures
    if (this->temperature_.has_value()) {
        for (const auto &[key, value] : this->temperature_.value()) {
            str += fmt::format("      temperature_{}:\n"
                               "        unit: \"°C\"\n"
                               "        values: [{}]\n",
                               key,
                               fmt::join(value, ", "));
        }
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_temperature_samples &samples) {
    std::string str{};

    append_map_values(str, "temperature_max", samples.get_temperature_max());

    str += fmt::format("temperature_psu: [{}]\n",
                       fmt::join(value_or_default(samples.get_temperature_psu()), ", "));

    append_map_values(str, "temperature", samples.get_temperature());

    // remove last newline
    str.pop_back();

    return out << str;
}

}  // namespace plssvm::detail::tracking
