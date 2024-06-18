/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_nvidia/nvml_samples.hpp"

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join
#include "nvml.h"        // NVML runtime functions

#include <cstddef>  // std::size_t
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <utility>  // std::forward
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

template <typename T, typename nvml_func, typename... Args>
[[nodiscard]] bool nvml_function_is_supported(nvml_func func, Args &&...args) {
    [[maybe_unused]] T val{};
    const nvmlReturn_t ret = func(std::forward<Args>(args)..., &val);
    return ret == NVML_SUCCESS;
}

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string nvml_general_samples::generate_yaml_string() const {
    nvmlDevice_t device{};
    nvmlDeviceGetHandleByIndex(static_cast<int>(this->get_device()), &device);

    std::string str{ "    general:\n" };

    // device name
    std::string device_name(NVML_DEVICE_NAME_V2_BUFFER_SIZE, '\0');
    const nvmlReturn_t ret = nvmlDeviceGetName(device, device_name.data(), device_name.size());
    if (ret == NVML_SUCCESS) {
        str += fmt::format("      name:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->name);
    }
    // persistence mode enabled
    if (nvml_function_is_supported<nvmlEnableState_t>(nvmlDeviceGetPersistenceMode, device)) {
        str += fmt::format("      persistence_mode:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->persistence_mode);
    }
    // number of cores
    if (nvml_function_is_supported<decltype(nvml_general_samples::num_cores)>(nvmlDeviceGetNumGpuCores, device)) {
        str += fmt::format("      num_cores:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_cores);
    }

    // performance state
    if (nvml_function_is_supported<nvmlPstates_t>(nvmlDeviceGetPerformanceState, device)) {
        str += fmt::format("      performance_state:\n"
                           "        unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_performance_state(), ", "));
    }
    // device compute and memory utilization
    if (nvml_function_is_supported<nvmlUtilization_t>(nvmlDeviceGetUtilizationRates, device)) {
        str += fmt::format("      utilization_gpu:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n"
                           "      utilization_mem:\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_utilization_gpu(), ", "),
                           fmt::join(this->get_utilization_mem(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_general_samples::nvml_general_sample &sample) {
    return out << fmt::format("performance_state: {}\n"
                              "utilization_gpu: {}\n"
                              "utilization_mem: {}",
                              sample.performance_state,
                              sample.utilization_gpu,
                              sample.utilization_mem);
}

std::ostream &operator<<(std::ostream &out, const nvml_general_samples &samples) {
    return out << fmt::format("name: {}\n"
                              "persistence_mode: {}\n"
                              "num_cores: {}\n"
                              "performance_state: [{}]\n"
                              "utilization_gpu: [{}]\n"
                              "utilization_mem: [{}]",
                              samples.name,
                              samples.persistence_mode,
                              samples.num_cores,
                              fmt::join(samples.get_performance_state(), ", "),
                              fmt::join(samples.get_utilization_gpu(), ", "),
                              fmt::join(samples.get_utilization_mem(), ", "));
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string nvml_clock_samples::generate_yaml_string() const {
    nvmlDevice_t device{};
    nvmlDeviceGetHandleByIndex(static_cast<int>(this->get_device()), &device);

    std::string str{ "    clock:\n" };

    // adaptive clock status
    if (nvml_function_is_supported<decltype(nvml_clock_samples::adaptive_clock_status)>(nvmlDeviceGetAdaptiveClockInfoStatus, device)) {
        str += fmt::format("      adaptive_clock_status:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->adaptive_clock_status == NVML_ADAPTIVE_CLOCKING_INFO_STATUS_ENABLED);
    }
    // maximum SM clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::clock_sm_max)>(nvmlDeviceGetMaxClockInfo, device, NVML_CLOCK_SM)) {
        str += fmt::format("      clock_sm_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_sm_max);
    }
    // minimum memory clock
    unsigned int clock_count{ 128 };
    std::vector<unsigned int> clocks(clock_count);
    nvmlReturn_t ret = nvmlDeviceGetSupportedMemoryClocks(device, &clock_count, clocks.data());
    if (ret == NVML_SUCCESS) {
        str += fmt::format("      clock_mem_min:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_mem_min);
    }
    // maximum memory clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::clock_mem_max)>(nvmlDeviceGetMaxClockInfo, device, NVML_CLOCK_MEM)) {
        str += fmt::format("      clock_mem_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_mem_max);
    }
    // minimum graph clock
    const unsigned int clock = clocks.front();
    clock_count = 128u;
    ret = nvmlDeviceGetSupportedGraphicsClocks(device, clock, &clock_count, clocks.data());
    if (ret == NVML_SUCCESS) {
        str += fmt::format("      clock_gpu_min:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_graph_min);
    }
    // maximum graph clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::clock_graph_max)>(nvmlDeviceGetMaxClockInfo, device, NVML_CLOCK_GRAPHICS)) {
        str += fmt::format("      clock_gpu_max:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->clock_graph_max);
    }

    // SM clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::nvml_clock_sample::clock_sm)>(nvmlDeviceGetClockInfo, device, NVML_CLOCK_SM)) {
        str += fmt::format("      clock_sm:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_clock_sm(), ", "));
    }
    // memory clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::nvml_clock_sample::clock_mem)>(nvmlDeviceGetClockInfo, device, NVML_CLOCK_MEM)) {
        str += fmt::format("      clock_mem:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_clock_mem(), ", "));
    }
    // graph clock
    if (nvml_function_is_supported<decltype(nvml_clock_samples::nvml_clock_sample::clock_graph)>(nvmlDeviceGetClockInfo, device, NVML_CLOCK_GRAPHICS)) {
        str += fmt::format("      clock_gpu:\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_clock_graph(), ", "));
    }
    // clock throttle reason
    if (nvml_function_is_supported<decltype(nvml_clock_samples::nvml_clock_sample::clock_throttle_reason)>(nvmlDeviceGetCurrentClocksThrottleReasons, device)) {
        str += fmt::format("      clock_throttle_reason:\n"
                           "        unit: \"bitmask\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_clock_throttle_reason(), ", "));
    }
    // clock is auto-boosted
    nvmlEnableState_t mode{};
    if (nvml_function_is_supported<nvmlEnableState_t>(nvmlDeviceGetAutoBoostedClocksEnabled, device, &mode)) {
        str += fmt::format("      auto_boosted_clocks:\n"
                           "        unit: \"bool\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_auto_boosted_clocks(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_clock_samples::nvml_clock_sample &sample) {
    return out << fmt::format("clock_graph: {}\n"
                              "clock_sm: {}\n"
                              "clock_mem: {}\n"
                              "clock_throttle_reason: {}\n"
                              "auto_boosted_clocks: {}",
                              sample.clock_graph,
                              sample.clock_sm,
                              sample.clock_mem,
                              sample.clock_throttle_reason,
                              sample.auto_boosted_clocks);
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
                              samples.adaptive_clock_status,
                              samples.clock_graph_min,
                              samples.clock_graph_max,
                              samples.clock_sm_max,
                              samples.clock_mem_min,
                              samples.clock_mem_max,
                              fmt::join(samples.get_clock_graph(), ", "),
                              fmt::join(samples.get_clock_sm(), ", "),
                              fmt::join(samples.get_clock_mem(), ", "),
                              fmt::join(samples.get_clock_throttle_reason(), ", "),
                              fmt::join(samples.get_auto_boosted_clocks(), ", "));
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string nvml_power_samples::generate_yaml_string() const {
    nvmlDevice_t device{};
    nvmlDeviceGetHandleByIndex(static_cast<int>(this->get_device()), &device);

    std::string str{ "    power:\n" };

    // power management limit
    if (nvml_function_is_supported<decltype(nvml_power_samples::power_management_limit)>(nvmlDeviceGetPowerManagementLimit, device)) {
        str += fmt::format("      power_management_limit:\n"
                           "        unit: \"mW\"\n"
                           "        values: {}\n",
                           this->power_management_limit);
    }
    // power enforced limit
    if (nvml_function_is_supported<decltype(nvml_power_samples::power_enforced_limit)>(nvmlDeviceGetEnforcedPowerLimit, device)) {
        str += fmt::format("      power_enforced_limit:\n"
                           "        unit: \"mW\"\n"
                           "        values: {}\n",
                           this->power_enforced_limit);
    }

    // power state
    if (nvml_function_is_supported<nvmlPstates_t>(nvmlDeviceGetPowerState, device)) {
        str += fmt::format("      power_state:\n"
                           "        unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_power_state(), ", "));
    }
    // current power usage
    if (nvml_function_is_supported<decltype(nvml_power_samples::nvml_power_sample::power_usage)>(nvmlDeviceGetPowerUsage, device)) {
        str += fmt::format("      power_usage:\n"
                           "        unit: \"mW\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_power_usage(), ", "));
    }
    // total energy consumed
    if (nvml_function_is_supported<decltype(nvml_power_samples::nvml_power_sample::power_total_energy_consumption)>(nvmlDeviceGetTotalEnergyConsumption, device)) {
        std::vector<decltype(nvml_power_samples::nvml_power_sample::power_total_energy_consumption)> consumed_energy(this->num_samples());
#pragma omp parallel for
        for (std::size_t i = 0; i < this->num_samples(); ++i) {
            consumed_energy[i] = this->get_power_total_energy_consumption()[i] - this->get_power_total_energy_consumption()[0];
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

std::ostream &operator<<(std::ostream &out, const nvml_power_samples::nvml_power_sample &sample) {
    return out << fmt::format("power_state: {}\n"
                              "power_usage: {}\n"
                              "power_total_energy_consumption: {}",
                              sample.power_state,
                              sample.power_usage,
                              sample.power_total_energy_consumption);
}

std::ostream &operator<<(std::ostream &out, const nvml_power_samples &samples) {
    return out << fmt::format("power_management_limit: {}\n"
                              "power_enforced_limit: {}\n"
                              "power_state: [{}]\n"
                              "power_usage: [{}]\n"
                              "power_total_energy_consumption: [{}]",
                              samples.power_management_limit,
                              samples.power_enforced_limit,
                              fmt::join(samples.get_power_state(), ", "),
                              fmt::join(samples.get_power_usage(), ", "),
                              fmt::join(samples.get_power_total_energy_consumption(), ", "));
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string nvml_memory_samples::generate_yaml_string() const {
    nvmlDevice_t device{};
    nvmlDeviceGetHandleByIndex(static_cast<int>(this->get_device()), &device);

    std::string str{ "    memory:\n" };

    // total memory size
    if (nvml_function_is_supported<nvmlMemory_t>(nvmlDeviceGetMemoryInfo, device)) {
        str += fmt::format("      memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->memory_total);
    }
    // maximum PCIe link speed
    if (nvml_function_is_supported<decltype(nvml_memory_samples::pcie_link_max_speed)>(nvmlDeviceGetPcieLinkMaxSpeed, device)) {
        str += fmt::format("      pcie_max_bandwidth:\n"
                           "        unit: \"MBPS\"\n"
                           "        values: {}\n",
                           this->pcie_link_max_speed);
    }
    // memory bus width
    if (nvml_function_is_supported<decltype(nvml_memory_samples::memory_bus_width)>(nvmlDeviceGetMemoryBusWidth, device)) {
        str += fmt::format("      memory_bus_width:\n"
                           "        unit: \"Bit\"\n"
                           "        values: {}\n",
                           this->memory_bus_width);
    }
    // maximum PCIe link generation
    if (nvml_function_is_supported<decltype(nvml_memory_samples::max_pcie_link_generation)>(nvmlDeviceGetGpuMaxPcieLinkGeneration, device)) {
        str += fmt::format("      max_pcie_link_generation:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->max_pcie_link_generation);
    }

    // free and used memory size
    if (nvml_function_is_supported<nvmlMemory_t>(nvmlDeviceGetMemoryInfo, device)) {
        str += fmt::format("      memory_free:\n"
                           "        unit \"B\"\n"
                           "        values: [{}]\n"
                           "      memory_used:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_memory_free(), ", "),
                           fmt::join(this->get_memory_used(), ", "));
    }
    // PCIe link speed
    if (nvml_function_is_supported<decltype(nvml_memory_samples::nvml_memory_sample::pcie_link_speed)>(nvmlDeviceGetPcieSpeed, device)) {
        str += fmt::format("      pcie_bandwidth:\n"
                           "        unit: \"MBPS\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_pcie_link_speed(), ", "));
    }
    // PCIe link width
    if (nvml_function_is_supported<decltype(nvml_memory_samples::nvml_memory_sample::pcie_link_width)>(nvmlDeviceGetCurrPcieLinkWidth, device)) {
        str += fmt::format("      pcie_link_width:\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_pcie_link_width(), ", "));
    }
    // PCIe link generation
    if (nvml_function_is_supported<decltype(nvml_memory_samples::nvml_memory_sample::pcie_link_generation)>(nvmlDeviceGetCurrPcieLinkGeneration, device)) {
        str += fmt::format("      pcie_link_generation:\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_pcie_link_generation(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_memory_samples::nvml_memory_sample &sample) {
    return out << fmt::format("memory_free: {}\n"
                              "memory_used: {}\n"
                              "pcie_link_speed: {}\n"
                              "pcie_link_width: {}\n"
                              "pcie_link_generation: {}",
                              sample.memory_free,
                              sample.memory_used,
                              sample.pcie_link_speed,
                              sample.pcie_link_width,
                              sample.pcie_link_generation);
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
                              samples.memory_total,
                              samples.pcie_link_max_speed,
                              samples.memory_bus_width,
                              samples.max_pcie_link_generation,
                              fmt::join(samples.get_memory_free(), ", "),
                              fmt::join(samples.get_memory_used(), ", "),
                              fmt::join(samples.get_pcie_link_speed(), ", "),
                              fmt::join(samples.get_pcie_link_width(), ", "),
                              fmt::join(samples.get_pcie_link_generation(), ", "));
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string nvml_temperature_samples::generate_yaml_string() const {
    nvmlDevice_t device{};
    nvmlDeviceGetHandleByIndex(static_cast<int>(this->get_device()), &device);

    std::string str{ "    temperature:\n" };

    // number of fans
    if (nvml_function_is_supported<decltype(nvml_temperature_samples::num_fans)>(nvmlDeviceGetNumFans, device)) {
        str += fmt::format("      num_fans:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_fans);
    }
    // min and max fan speed
    if (this->num_fans > 0) {
        unsigned int min_fan_speed_test{};
        if (nvml_function_is_supported<decltype(nvml_temperature_samples::min_fan_speed)>(nvmlDeviceGetMinMaxFanSpeed, device, &min_fan_speed_test)) {
            str += fmt::format("      min_fan_speed:\n"
                               "        unit: \"percentage\"\n"
                               "        values: {}\n"
                               "      max_fan_speed:\n"
                               "        unit: \"percentage\"\n"
                               "        values: {}\n",
                               this->min_fan_speed,
                               this->max_fan_speed);
        }
    }
    // temperature threshold GPU max
    if (nvml_function_is_supported<decltype(nvml_temperature_samples::temperature_threshold_gpu_max)>(nvmlDeviceGetTemperatureThreshold, device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX)) {
        str += fmt::format("      temperature_gpu_max:\n"
                           "        unit: \"°C\"\n"
                           "        values: {}\n",
                           this->temperature_threshold_gpu_max);
    }
    // temperature threshold memory max
    if (nvml_function_is_supported<decltype(nvml_temperature_samples::temperature_threshold_mem_max)>(nvmlDeviceGetTemperatureThreshold, device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX)) {
        str += fmt::format("      temperature_mem_max:\n"
                           "        unit: \"°C\"\n"
                           "        values: {}\n",
                           this->temperature_threshold_mem_max);
    }

    // fan speed
    if (nvml_function_is_supported<decltype(nvml_temperature_samples::nvml_temperature_sample::fan_speed)>(nvmlDeviceGetFanSpeed, device)) {
        str += fmt::format("      fan_speed:\n"
                           "        unit \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_fan_speed(), ", "));
    }
    // temperature GPU
    if (nvml_function_is_supported<decltype(nvml_temperature_samples::nvml_temperature_sample::temperature_gpu)>(nvmlDeviceGetTemperature, device, NVML_TEMPERATURE_GPU)) {
        str += fmt::format("      temperature_gpu:\n"
                           "        unit: \"°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->get_temperature_gpu(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples::nvml_temperature_sample &sample) {
    return out << fmt::format("fan_speed: {}\n"
                              "temperature_gpu: {}",
                              sample.fan_speed,
                              sample.temperature_gpu);
}

std::ostream &operator<<(std::ostream &out, const nvml_temperature_samples &samples) {
    return out << fmt::format("num_fans: {}\n"
                              "min_fan_speed: {}\n"
                              "max_fan_speed: {}\n"
                              "temperature_threshold_gpu_max: {}\n"
                              "temperature_threshold_mem_max: {}\n"
                              "fan_speed: [{}]\n"
                              "temperature_gpu: [{}]",
                              samples.num_fans,
                              samples.min_fan_speed,
                              samples.max_fan_speed,
                              samples.temperature_threshold_gpu_max,
                              samples.temperature_threshold_mem_max,
                              fmt::join(samples.get_fan_speed(), ", "),
                              fmt::join(samples.get_temperature_gpu(), ", "));
}

}  // namespace plssvm::detail::tracking
