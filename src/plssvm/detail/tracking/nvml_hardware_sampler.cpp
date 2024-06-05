/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/nvml_hardware_sampler.hpp"

#include "plssvm/detail/tracking/time_point.hpp"  // plssvm::detail::tracking::time_point

#include "nvml.h"
#include <cstdint>    // std::uint64_t
#include <exception>  // std::exception, std::terminate
#include <iostream>   // std::cerr, std::endl
#include <mutex>      // std::call_once

namespace plssvm::detail::tracking {

#define PLSSVM_NVML_ERROR_CHECK(errc)                                                                       \
    if ((errc) != NVML_SUCCESS && (errc) != NVML_ERROR_NOT_SUPPORTED) {                                     \
        throw std::runtime_error{ std::string{ "Error in NVML function call: " } + nvmlErrorString(errc) }; \
    }

nvml_hardware_sampler::nvml_hardware_sampler(const int device_id, const unsigned long long sampling_interval) :
    base_type{ sampling_interval } {
    // make sure that nvmlInit is only called once for all instances
    std::call_once(nvml_init_once_, []() {
        nvmlInit();
    });
    ++instances_;

    // get the device handle for which this hardware sampler is responsible for
    PLSSVM_NVML_ERROR_CHECK((nvmlDeviceGetHandleByIndex(device_id, &device_)));
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

// struct sample_type {
//     time_point_type time;
//
//     // clock related information
//     double clock_graph = 0.0;
//     double clock_sm = 0.0;
//     double clock_mem = 0.0;
//     std::uint64_t clock_throttle_reason = 0;
//     double clock_graph_max = 0.0;
//     double clock_sm_max = 0.0;
//     double clock_mem_max = 0.0;
//
//     // temperature related information
//     std::int64_t fan_speed = 0;
//     std::int64_t temperature_gpu = 0;
//     std::int64_t temperature_threshold_gpu_max = 0;
//     std::int64_t temperature_threshold_mem_max = 0;
//
//     // memory related information
//     double memory_free = 0.0;
//     double memory_used = 0.0;
//     double memory_total = 0.0;
//
//     // power related information
//     std::int64_t power_state = 0;
//     double power_usage = 0.0;
//     double power_limit = 0.0;
//     double power_default_limit = 0.0;
//     double power_total_energy_consumption = 0.0;
//
//     // general information
//     std::int64_t performance_state = 0;
//     std::uint32_t utilization_gpu = 0;
//     std::uint32_t utilization_mem = 0;
// };

sample_type nvml_hardware_sampler::get_sample_measurement() {
    sample_type sample{};

    // set timestamp
    sample.time = clock_type::now();

    // retrieve clock related information
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device_, NVML_CLOCK_GRAPHICS, &sample.clock_graph));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device_, NVML_CLOCK_SM, &sample.clock_sm));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetClockInfo(device_, NVML_CLOCK_MEM, &sample.clock_mem));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetCurrentClocksThrottleReasons(device_, &sample.clock_throttle_reason));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_GRAPHICS, &sample.clock_graph_max));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_SM, &sample.clock_sm_max));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMaxClockInfo(device_, NVML_CLOCK_MEM, &sample.clock_mem_max));

    // retrieve temperature related information
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetFanSpeed(device_, &sample.fan_speed));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, &sample.temperature_gpu));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device_, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &sample.temperature_threshold_gpu_max));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTemperatureThreshold(device_, NVML_TEMPERATURE_THRESHOLD_MEM_MAX, &sample.temperature_threshold_mem_max));

    // retrieve memory related information
    nvmlMemory_t memory_info{};
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetMemoryInfo(device_, &memory_info));
    sample.memory_free = memory_info.free;
    sample.memory_used = memory_info.used;
    sample.memory_total = memory_info.total;

    // retrieve power related information
    nvmlPstates_t pstate{};
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPerformanceState(device_, &pstate));
    sample.power_state = static_cast<int>(pstate);
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerUsage(device_, &sample.power_usage));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPowerManagementLimit(device_, &sample.power_default_limit));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetEnforcedPowerLimit(device_, &sample.power_limit));
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetEnforcedPowerLimit(device_, &sample.power_limit));
    sample.power_total_energy_consumption = static_cast<unsigned long long>(this->get_total_energy_consumption()) - power_total_energy_consumption_start_;

    // retrieve general infromation
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetPerformanceState(device_, &pstate));
    sample.performance_state = static_cast<int>(pstate);
    nvmlUtilization_t util{};
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetUtilizationRates(device_, &util));
    sample.utilization_gpu = util.gpu;
    sample.utilization_mem = util.memory;

    return sample;
}

std::uint64_t nvml_hardware_sampler::get_total_energy_consumption() {
    unsigned long long energy{ 0 };
    PLSSVM_NVML_ERROR_CHECK(nvmlDeviceGetTotalEnergyConsumption(device_, &energy));
    return static_cast<std::uint64_t>(energy);
}

}  // namespace plssvm::detail::tracking
