/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/nvml_samples.hpp"

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join
#include "nvml.h"        // NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED

#include <cstddef>  // std::size_t
#include <ostream>  // std::ostream
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples) {
    return out << fmt::format("      general:\n"
                              "        name:\n"
                              "          unit: \"string\"\n"
                              "          values: \"{}\"\n"
                              "        persistence_mode:\n"
                              "          unit: \"bool\"\n"
                              "          values: {}\n"
                              "        num_cores:\n"
                              "          unit: \"int\"\n"
                              "          values: {}\n"
                              "        performance_state:\n"
                              "          unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                              "          values: [{}]\n"
                              "        utilization_gpu:\n"
                              "          unit: \"percentage\"\n"
                              "          values: [{}]\n"
                              "        utilization_mem:\n"
                              "          unit: \"percentage\"\n"
                              "          values: [{}]",
                              samples.name,
                              samples.persistence_mode,
                              samples.num_cores,
                              fmt::join(samples.get_performance_state(), ", "),
                              fmt::join(samples.get_utilization_gpu(), ", "),
                              fmt::join(samples.get_utilization_mem(), ", "));
}

std::ostream &operator<<(std::ostream &out, const nvml_clock_samples &samples) {
    return out << fmt::format("      clock:\n"
                              "        adaptive_clock_status:\n"
                              "          unit: \"bool\"\n"
                              "          values: {}\n"
                              "        clock_graph_max:\n"
                              "          unit: \"MHz\"\n"
                              "          values: {}\n"
                              "        clock_sm_max:\n"
                              "          unit: \"MHz\"\n"
                              "          values: {}\n"
                              "        clock_mem_max:\n"
                              "          unit: \"MHz\"\n"
                              "          values: {}\n"
                              "        clock_graph:\n"
                              "          unit: \"MHz\"\n"
                              "          values: [{}]\n"
                              "        clock_sm:\n"
                              "          unit: \"MHz\"\n"
                              "          values: [{}]\n"
                              "        clock_mem:\n"
                              "          unit: \"MHz\"\n"
                              "          values: [{}]\n"
                              "        clock_throttle_reason:\n"
                              "          unit: \"bitmask\"\n"
                              "          values: [{}]\n"
                              "        auto_boosted_clocks:\n"
                              "          unit: \"bool\"\n"
                              "          values: [{}]",
                              fmt::format("{}", samples.adaptive_clock_status == NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED),
                              samples.clock_graph_max,
                              samples.clock_sm_max,
                              samples.clock_mem_max,
                              fmt::join(samples.get_clock_graph(), ", "),
                              fmt::join(samples.get_clock_sm(), ", "),
                              fmt::join(samples.get_clock_mem(), ", "),
                              fmt::join(samples.get_clock_throttle_reason(), ", "),
                              fmt::join(samples.get_auto_boosted_clocks(), ", "));
}

std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples) {
    std::vector<unsigned long long> consumed_energy(samples.num_samples());
#pragma omp parallel for
    for (std::size_t i = 0; i < samples.num_samples(); ++i) {
        consumed_energy[i] = samples.get_power_total_energy_consumption()[i] - samples.get_power_total_energy_consumption()[0];
    }
    return out << fmt::format("      power:\n"
                              "        power_management_limit:\n"
                              "          unit: \"mW\"\n"
                              "          values: {}\n"
                              "        power_enforced_limit:\n"
                              "          unit: \"mW\"\n"
                              "          values: {}\n"
                              "        power_state:\n"
                              "          unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                              "          values: [{}]\n"
                              "        power_usage:\n"
                              "          unit: \"mW\"\n"
                              "          values: [{}]\n"
                              "        power_total_energy_consumed:\n"
                              "          unit: \"J\"\n"
                              "          values: [{}]",
                              samples.power_management_limit,
                              samples.power_enforced_limit,
                              fmt::join(samples.get_power_state(), ", "),
                              fmt::join(samples.get_power_usage(), ", "),
                              fmt::join(consumed_energy, ", "));
}

std::ostream &operator<<(std::ostream &out, const nvml_memory_samples &samples) {
    return out << fmt::format("      memory:\n"
                              "        memory_total:\n"
                              "          unit: \"B\"\n"
                              "          values: {}\n"
                              "        memory_bus_width:\n"
                              "          unit: \"Bit\"\n"
                              "          values: {}\n"
                              "        max_pcie_link_generation:\n"
                              "          unit: \"int\"\n"
                              "          values: {}\n"
                              "        pcie_link_max_speed:\n"
                              "          unit: \"MBPS\"\n"
                              "          values: {}\n"
                              "        memory_free:\n"
                              "          unit \"B\"\n"
                              "          values: [{}]\n"
                              "        memory_used:\n"
                              "          unit: \"B\"\n"
                              "          values: [{}]\n"
                              "        pcie_link_width:\n"
                              "          unit: \"int\"\n"
                              "          values: [{}]\n"
                              "        pcie_link_generation:\n"
                              "          unit: \"int\"\n"
                              "          values: [{}]",
                              samples.memory_total,
                              samples.memory_bus_width,
                              samples.max_pcie_link_generation,
                              samples.pcie_link_max_speed,
                              fmt::join(samples.get_memory_free(), ", "),
                              fmt::join(samples.get_memory_used(), ", "),
                              fmt::join(samples.get_pcie_link_width(), ", "),
                              fmt::join(samples.get_pcie_link_generation(), ", "));
}

std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples &samples) {
    return out << fmt::format("      temperature:\n"
                              "        num_fans:\n"
                              "          unit: \"int\"\n"
                              "          values: {}\n"
                              "        min_fan_speed:\n"
                              "          unit: \"percentage\"\n"
                              "          values: {}\n"
                              "        max_fan_speed:\n"
                              "          unit: \"percentage\"\n"
                              "          values: {}\n"
                              "        temperature_threshold_gpu_max:\n"
                              "          unit: \"°C\"\n"
                              "          values: {}\n"
                              "        temperature_threshold_mem_max:\n"
                              "          unit: \"°C\"\n"
                              "          values: {}\n"
                              "        fan_speed:\n"
                              "          unit \"percentage\"\n"
                              "          values: [{}]\n"
                              "        temperature_gpu:\n"
                              "          unit: \"°C\"\n"
                              "          values: [{}]",
                              samples.num_fans,
                              samples.min_fan_speed,
                              samples.max_fan_speed,
                              samples.temperature_threshold_gpu_max,
                              samples.temperature_threshold_mem_max,
                              fmt::join(samples.get_fan_speed(), ", "),
                              fmt::join(samples.get_temperature_gpu(), ", "));
}

}  // namespace plssvm::detail::tracking
