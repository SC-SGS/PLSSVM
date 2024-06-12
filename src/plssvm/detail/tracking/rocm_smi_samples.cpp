/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/rocm_smi_samples.hpp"

#include "plssvm/detail/operators.hpp"  // operators namespace

#include "fmt/core.h"           // fmt::format
#include "fmt/format.h"         // fmt::join
#include "rocm_smi/rocm_smi.h"  // ROCm SMI runtime functions

#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint32_t
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

template <typename T, typename rsmi_func, typename... Args>
[[nodiscard]] bool rsmi_function_is_supported(rsmi_func func, Args &&...args) {
    [[maybe_unused]] T val{};
    const rsmi_status_t ret = func(std::forward<Args>(args)..., &val);
    return ret == RSMI_STATUS_SUCCESS;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_general_samples &samples) {
    const std::uint32_t device = samples.get_device();

    std::string str{ "      general:\n" };

    // device name
    std::string name(static_cast<std::string::size_type>(1024), '\0');
    const rsmi_status_t ret = rsmi_dev_name_get(device, name.data(), name.size());
    if (ret == RSMI_STATUS_SUCCESS) {
        str += fmt::format("        name:\n"
                           "          unit: \"string\"\n"
                           "          values: \"{}\"\n",
                           samples.name);
    }

    // performance state
    if (rsmi_function_is_supported<rsmi_dev_perf_level_t>(rsmi_dev_perf_level_get, device)) {
        str += fmt::format("        performance_state:\n"
                           "          unit: \"int - see rsmi_dev_perf_level_t\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_performance_state(), ", "));
    }
    // device compute utilization
    if (rsmi_function_is_supported<decltype(rocm_smi_general_samples::rocm_smi_general_sample::utilization_gpu)>(rsmi_dev_busy_percent_get, device)) {
        str += fmt::format("        utilization_gpu:\n"
                           "          unit: \"percentage\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_utilization_gpu(), ", "));
    }
    // device memory utilization
    if (rsmi_function_is_supported<decltype(rocm_smi_general_samples::rocm_smi_general_sample::utilization_mem)>(rsmi_dev_memory_busy_percent_get, device)) {
        str += fmt::format("        utilization_mem:\n"
                           "          unit: \"percentage\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_utilization_mem(), ", "));
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_clock_samples &samples) {
    const std::uint32_t device = samples.get_device();

    std::string str{ "      clock:\n" };
    // socket clock min/max frequencies
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_SOC)) {
        str += fmt::format("        clock_socket_min:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n"
                           "        clock_socket_max:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n",
                           samples.clock_socket_min,
                           samples.clock_socket_max);
    }
    // memory clock min/max frequencies
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_MEM)) {
        str += fmt::format("        clock_memory_min:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n"
                           "        clock_memory_max:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n",
                           samples.clock_memory_min,
                           samples.clock_memory_max);
    }

    // system clock min/max frequencies
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_SYS)) {
        str += fmt::format("        clock_gpu_min:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n"
                           "        clock_gpu_max:\n"
                           "          unit: \"Hz\"\n"
                           "          values: {}\n",
                           samples.clock_system_min,
                           samples.clock_system_max);
    }

    // socket clock frequency
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_SOC)) {
        str += fmt::format("        clock_socket:\n"
                           "          unit: \"Hz\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_clock_socket(), ", "));
    }
    // memory clock frequency
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_MEM)) {
        str += fmt::format("        clock_memory:\n"
                           "          unit: \"Hz\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_clock_memory(), ", "));
    }
    // system clock frequency
    if (rsmi_function_is_supported<rsmi_frequencies_t>(rsmi_dev_gpu_clk_freq_get, device, RSMI_CLK_TYPE_SYS)) {
        str += fmt::format("        clock_gpu:\n"
                           "          unit: \"Hz\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_clock_system(), ", "));
    }
    // clock throttle reason
    if (rsmi_function_is_supported<decltype(rocm_smi_clock_samples::rocm_smi_clock_sample::clock_throttle_reason)>(rsmi_dev_metrics_throttle_status_get, device)) {
        str += fmt::format("        clock_throttle_reason:\n"
                           "          unit: \"bitmask\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_clock_throttle_reason(), ", "));
    }
    // overdrive level
    if (rsmi_function_is_supported<decltype(rocm_smi_clock_samples::rocm_smi_clock_sample::overdrive_level)>(rsmi_dev_overdrive_level_get, device)) {
        str += fmt::format("        overdrive_level:\n"
                           "          unit: \"percentage\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_overdrive_level(), ", "));
    }
    // memory overdrive level
    if (rsmi_function_is_supported<decltype(rocm_smi_clock_samples::rocm_smi_clock_sample::memory_overdrive_level)>(rsmi_dev_mem_overdrive_level_get, device)) {
        str += fmt::format("        memory_overdrive_level:\n"
                           "          unit: \"percentage\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_memory_overdrive_level(), ", "));
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_power_samples &samples) {
    const std::uint32_t device = samples.get_device();

    std::string str{ "      power:\n" };

    // default power cap
    if (rsmi_function_is_supported<decltype(rocm_smi_power_samples::power_default_cap)>(rsmi_dev_power_cap_default_get, device)) {
        str += fmt::format("        power_management_limit:\n"
                           "          unit: \"muW\"\n"
                           "          values: {}\n",
                           samples.power_default_cap);
    }
    // power cap
    if (rsmi_function_is_supported<decltype(rocm_smi_power_samples::power_cap)>(rsmi_dev_power_cap_get, device, std::uint32_t{ 0 })) {
        str += fmt::format("        power_enforced_limit:\n"
                           "          unit: \"muW\"\n"
                           "          values: {}\n",
                           samples.power_cap);
    }
    // power measurement type
    std::uint64_t power_usage{};
    if (rsmi_function_is_supported<RSMI_POWER_TYPE>(rsmi_dev_power_get, device, &power_usage)) {
        str += fmt::format("        power_measurement_type:\n"
                           "          unit: \"string\"\n"
                           "          values: {}\n",
                           samples.power_type);
    }

    // current power usage
    if (rsmi_function_is_supported<RSMI_POWER_TYPE>(rsmi_dev_power_get, device, &power_usage)) {
        str += fmt::format("        power_usage:\n"
                           "          unit: \"muW\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_power_usage(), ", "));
    }
    // total energy consumed
    [[maybe_unused]] decltype(rocm_smi_power_samples::rocm_smi_power_sample::power_total_energy_consumption) total_consumed_power{};
    [[maybe_unused]] float resolution{};
    if (rsmi_function_is_supported<std::uint64_t>(rsmi_dev_energy_count_get, device, &total_consumed_power, &resolution)) {
        std::vector<decltype(rocm_smi_power_samples::rocm_smi_power_sample::power_total_energy_consumption)> consumed_energy(samples.num_samples());
#pragma omp parallel for
        for (std::size_t i = 0; i < samples.num_samples(); ++i) {
            consumed_energy[i] = samples.get_power_total_energy_consumption()[i] - samples.get_power_total_energy_consumption()[0];
        }
        str += fmt::format("        power_total_energy_consumed:\n"
                           "          unit: \"muJ\"\n"
                           "          values: [{}]\n",
                           fmt::join(consumed_energy, ", "));
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_memory_samples &samples) {
    const std::uint32_t device = samples.get_device();

    std::string str{ "      memory:\n" };

    // total memory
    if (rsmi_function_is_supported<decltype(rocm_smi_memory_samples::memory_total)>(rsmi_dev_memory_total_get, device, RSMI_MEM_TYPE_VRAM)) {
        str += fmt::format("        memory_total:\n"
                           "          unit: \"B\"\n"
                           "          values: {}\n",
                           samples.memory_total);
    }
    // total visible memory
    if (rsmi_function_is_supported<decltype(rocm_smi_memory_samples::memory_total)>(rsmi_dev_memory_total_get, device, RSMI_MEM_TYPE_VIS_VRAM)) {
        str += fmt::format("        visible_memory_total:\n"
                           "          unit: \"B\"\n"
                           "          values: {}\n",
                           samples.visible_memory_total);
    }
    // min/max number of PCIe lanes
    if (rsmi_function_is_supported<rsmi_pcie_bandwidth_t>(rsmi_dev_pci_bandwidth_get, device)) {
        str += fmt::format("        min_num_pcie_lanes:\n"
                           "          unit: \"int\"\n"
                           "          values: {}\n"
                           "        max_num_pcie_lanes:\n"
                           "          unit: \"int\"\n"
                           "          values: {}\n",
                           samples.min_num_pcie_lanes,
                           samples.max_num_pcie_lanes);
    }

    // used and free memory
    if (rsmi_function_is_supported<decltype(rocm_smi_memory_samples::rocm_smi_memory_sample::memory_used)>(rsmi_dev_memory_usage_get, device, RSMI_MEM_TYPE_VRAM)) {
        using namespace plssvm::operators;
        std::vector<decltype(rocm_smi_memory_samples::rocm_smi_memory_sample::memory_used)> memory_free(samples.num_samples(), samples.memory_total);
        memory_free -= samples.get_memory_used();
        str += fmt::format("        memory_free:\n"
                           "          unit: \"B\"\n"
                           "          values: [{}]\n"
                           "        memory_used:\n"
                           "          unit: \"B\"\n"
                           "          values: [{}]\n",
                           fmt::join(memory_free, ", "),
                           fmt::join(samples.get_memory_used(), ", "));
    }
    // number of PCIe lanes and bandwidth
    if (rsmi_function_is_supported<rsmi_pcie_bandwidth_t>(rsmi_dev_pci_bandwidth_get, device)) {
        str += fmt::format("        pcie_bandwidth:\n"
                           "          unit: \"T/s\"\n"
                           "          values: [{}]\n"
                           "        pcie_num_lanes:\n"
                           "          unit: \"int\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_pcie_transfer_rate(), ", "),
                           fmt::join(samples.get_num_pcie_lanes(), ", "));
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

std::ostream &operator<<(std::ostream &out, const rocm_smi_temperature_samples &samples) {
    const std::uint32_t device = samples.get_device();

    std::string str{ "      temperature:\n" };

    // number of fans (emulated)
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::rocm_smi_temperature_sample::fan_speed)>(rsmi_dev_fan_speed_get, device, std::uint32_t{ 0 })) {
        str += fmt::format("        num_fans:\n"
                           "          unit: \"int\"\n"
                           "          values: {}\n",
                           samples.num_fans);
    }
    // maximum fan speed
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::max_fan_speed)>(rsmi_dev_fan_speed_max_get, device, std::uint32_t{ 0 })) {
        str += fmt::format("        max_fan_speed:\n"
                           "          unit: \"int\"\n"
                           "          values: {}\n",
                           samples.max_fan_speed);
    }
    // minimum GPU edge temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_edge_min)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MIN)) {
        str += fmt::format("        temperature_gpu_min:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_edge_min);
    }
    // maximum GPU edge temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_edge_max)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_MAX)) {
        str += fmt::format("        temperature_gpu_max:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_edge_max);
    }
    // minimum GPU hotspot temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_hotspot_min)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MIN)) {
        str += fmt::format("        temperature_hotspot_min:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_hotspot_min);
    }
    // maximum GPU hotspot temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_hotspot_max)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_MAX)) {
        str += fmt::format("        temperature_hotspot_max:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_hotspot_max);
    }
    // minimum GPU memory temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_memory_min)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MIN)) {
        str += fmt::format("        temperature_memory_min:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_memory_min);
    }
    // maximum GPU memory temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::temperature_memory_max)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_MAX)) {
        str += fmt::format("        temperature_memory_max:\n"
                           "          unit: \"m°C\"\n"
                           "          values: {}\n",
                           samples.temperature_memory_max);
    }

    // fan speed
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::rocm_smi_temperature_sample::fan_speed)>(rsmi_dev_fan_speed_get, device, std::uint32_t{ 0 })) {
        std::vector<double> fan_speed_percent(samples.num_samples());
#pragma omp parallel for
        for (std::size_t i = 0; i < fan_speed_percent.size(); ++i) {
            fan_speed_percent[i] = static_cast<double>(samples.get_fan_speed()[i]) / static_cast<double>(RSMI_MAX_FAN_SPEED);
        }
        str += fmt::format("        fan_speed:\n"
                           "          unit: \"percentage\"\n"
                           "          values: [{}]\n",
                           fmt::join(fan_speed_percent, ", "));
    }
    // GPU edge temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::rocm_smi_temperature_sample::temperature_edge)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT)) {
        str += fmt::format("        temperature_gpu:\n"
                           "          unit: \"m°C\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_temperature_edge(), ", "));
    }
    // GPU hotspot temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::rocm_smi_temperature_sample::temperature_hotspot)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT)) {
        str += fmt::format("        temperature_hotspot:\n"
                           "          unit: \"m°C\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_temperature_hotspot(), ", "));
    }
    // GPU memory temperature
    if (rsmi_function_is_supported<decltype(rocm_smi_temperature_samples::rocm_smi_temperature_sample::temperature_memory)>(rsmi_dev_temp_metric_get, device, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT)) {
        str += fmt::format("        temperature_memory:\n"
                           "          unit: \"m°C\"\n"
                           "          values: [{}]\n",
                           fmt::join(samples.get_temperature_memory(), ", "));
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

}  // namespace plssvm::detail::tracking
