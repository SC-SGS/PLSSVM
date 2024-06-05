/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/nvml_hardware_sampler.hpp"

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/nvml_samples.hpp"

#include "nvml.h"

#include "fmt/chrono.h"
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join

#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <exception>  // std::exception, std::terminate
#include <iostream>   // std::cerr, std::endl
#include <mutex>      // std::call_once
#include <vector>     // std::vector

namespace plssvm::detail::tracking {

#define PLSSVM_NVML_ERROR_CHECK(errc)                                                                       \
    if ((errc) != NVML_SUCCESS && (errc) != NVML_ERROR_NOT_SUPPORTED) {                                     \
        throw std::runtime_error{ std::string{ "Error in NVML function call: " } + nvmlErrorString(errc) }; \
    }

nvmlDevice_t device_id_to_nvml_handle(const std::size_t device_id) {
    // get the device handle for which this hardware sampler is responsible for
    nvmlDevice_t device{};
    PLSSVM_NVML_ERROR_CHECK((nvmlDeviceGetHandleByIndex(static_cast<int>(device_id), &device)));
    return device;
}

nvml_hardware_sampler::nvml_hardware_sampler(const std::size_t device_id, const unsigned long long sampling_interval) :  // TODO: chrono!
    hardware_sampler{ sampling_interval },
    device_id_{ device_id } {
    // make sure that nvmlInit is only called once for all instances
    std::call_once(nvml_init_once_, []() {
        nvmlInit();
    });
    ++instances_;
}

nvml_hardware_sampler::~nvml_hardware_sampler() {
    try {
        // the last instance must shut down the NVML runtime
        if (--instances_ == 0) {
            // make sure that nvmlShutdown is only called once
            std::call_once(nvml_shutdown_once_, []() { nvmlShutdown(); });  // TODO: can't create new nvml_hardware_sampler instance afterwards?!
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
}

std::string nvml_hardware_sampler::assemble_yaml_sample_string() const {
    std::string str{ "\n" };

    // TODO: zero or single elements!

    // format time points
    str += fmt::format("    time_points: [{}]\n", fmt::join(general_samples_.get_times_since_start(), ", "));

    // format general information
    str += fmt::format("    performance_state:\n"
                       "      unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                       "      values: [{}]\n"
                       "    utilization_gpu:\n"
                       "      unit: \"percentage\"\n"
                       "      values: [{}]\n"
                       "    utilization_mem:\n"
                       "      unit: \"percentage\"\n"
                       "      values: [{}]\n",
                       fmt::join(general_samples_.get_performance_states(), ", "),
                       fmt::join(general_samples_.get_utilizations_gpu(), ", "),
                       fmt::join(general_samples_.get_utilizations_mem(), ", "));

    // format clock related information
    str += fmt::format("    clock_graph:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n"
                       "    clock_sm:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n"
                       "    clock_mem:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n"
                       "    clock_throttle_reason:\n"
                       "      unit: \"bitmask\"\n"
                       "      values: [{}]\n"
                       "    clock_graph_max:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n"
                       "    clock_sm_max:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n"
                       "    clock_mem_max:\n"
                       "      unit: \"MHz\"\n"
                       "      values: [{}]\n",
                       fmt::join(clock_samples_.get_clocks_graph(), ", "),
                       fmt::join(clock_samples_.get_clocks_sm(), ", "),
                       fmt::join(clock_samples_.get_clocks_mem(), ", "),
                       fmt::join(clock_samples_.get_clocks_throttle_reason(), ", "),
                       fmt::join(clock_samples_.get_clocks_graph_max(), ", "),
                       fmt::join(clock_samples_.get_clocks_sm_max(), ", "),
                       fmt::join(clock_samples_.get_clocks_mem_max(), ", "));

    // format power related information
    std::vector<unsigned long long> consumed_energy(power_samples_.num_samples());
#pragma omp parallel for
    for (std::size_t i = 0; i < power_samples_.num_samples(); ++i) {
        consumed_energy[i] = power_samples_.get_power_total_energy_consumptions()[i] - power_samples_.get_power_total_energy_consumptions()[0];
    }
    str += fmt::format("    power_state:\n"
                       "      unit: \"0 - maximum performance; 15 - minimum performance; 32 - unknown\"\n"
                       "      values: [{}]\n"
                       "    power_usage:\n"
                       "      unit: \"mW\"\n"
                       "      values: [{}]\n"
                       "    power_management_limit:\n"
                       "      unit: \"mW\"\n"
                       "      values: [{}]\n"
                       "    power_enforced_limit:\n"
                       "      unit: \"mW\"\n"
                       "      values: [{}]\n"
                       "    power_total_energy_consumed:\n"
                       "      unit: \"J\"\n"
                       "      values: [{}]\n",
                       fmt::join(power_samples_.get_power_states(), ", "),
                       fmt::join(power_samples_.get_power_usages(), ", "),
                       fmt::join(power_samples_.get_power_management_limits(), ", "),
                       fmt::join(power_samples_.get_power_enforced_limits(), ", "),
                       fmt::join(consumed_energy, ", "));

    // format memory related information
    str += fmt::format("    memory_free:\n"
                       "      unit \"B\"\n"
                       "      values: [{}]\n"
                       "    memory_used:\n"
                       "      unit: \"B\"\n"
                       "      values: [{}]\n"
                       "    memory_total:\n"
                       "      unit: \"B\"\n"
                       "      values: [{}]\n",
                       fmt::join(memory_samples_.get_memory_free(), ", "),
                       fmt::join(memory_samples_.get_memory_used(), ", "),
                       fmt::join(memory_samples_.get_memory_total(), ", "));

    // format temperature related information
    str += fmt::format("    fan_speed:\n"
                       "      unit \"percentage\"\n"
                       "      values: [{}]\n"
                       "    temperature_gpu:\n"
                       "      unit: \"°C\"\n"
                       "      values: [{}]\n"
                       "    temperature_threshold_gpu_max:\n"
                       "      unit: \"°C\"\n"
                       "      values: [{}]\n"
                       "    temperature_threshold_mem_max:\n"
                       "      unit: \"°C\"\n"
                       "      values: [{}]\n",
                       fmt::join(temperature_samples_.get_fan_speeds(), ", "),
                       fmt::join(temperature_samples_.get_temperatures_gpu(), ", "),
                       fmt::join(temperature_samples_.get_temperatures_threshold_gpu_max(), ", "),
                       fmt::join(temperature_samples_.get_temperatures_threshold_mem_max(), ", "));

    // remove last newline
    str.pop_back();
    return str;
}

void nvml_hardware_sampler::add_sample() {
    // get the nvml handle from the device_id
    nvmlDevice_t device = device_id_to_nvml_handle(device_id_);

    // retrieve general information
    {
        nvml_general_samples::nvml_general_sample sample{};
        sample.time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);
        nvmlPstates_t pstate{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPerformanceState(device, &pstate));
        sample.performance_state = static_cast<int>(pstate);
        nvmlUtilization_t util{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetUtilizationRates(device, &util));
        sample.utilization_gpu = util.gpu;
        sample.utilization_mem = util.memory;
        this->general_samples_.add_sample(sample);
    }
    // retrieve clock related information
    {
        nvml_clock_samples::nvml_clock_sample sample{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &sample.clock_graph));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sample.clock_sm));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &sample.clock_mem));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrentClocksThrottleReasons(device, &sample.clock_throttle_reason));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &sample.clock_graph_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &sample.clock_sm_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_MEM, &sample.clock_mem_max));
        this->clock_samples_.add_sample(sample);
    }
    // retrieve power related information
    {
        nvml_power_samples::nvml_power_sample sample{};
        nvmlPstates_t pstate{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerState(device, &pstate));
        sample.power_state = static_cast<int>(pstate);
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerUsage(device, &sample.power_usage));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerManagementLimit(device, &sample.power_management_limit));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetEnforcedPowerLimit(device, &sample.power_enforced_limit));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTotalEnergyConsumption(device, &sample.power_total_energy_consumption));
        this->power_samples_.add_sample(sample);
    }
    // retrieve memory related information
    {
        nvml_memory_samples::nvml_memory_sample sample{};
        nvmlMemory_t memory_info{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryInfo(device, &memory_info));
        sample.memory_free = memory_info.free;
        sample.memory_used = memory_info.used;
        sample.memory_total = memory_info.total;
        this->memory_samples_.add_sample(sample);
    }
    // retrieve temperature related information
    {
        nvml_temperature_samples::nvml_temperature_sample sample{};
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetFanSpeed(device, &sample.fan_speed));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &sample.temperature_gpu));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &sample.temperature_threshold_gpu_max));
        PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, &sample.temperature_threshold_mem_max));
        this->temperature_samples_.add_sample(sample);
    }
}

}  // namespace plssvm::detail::tracking
