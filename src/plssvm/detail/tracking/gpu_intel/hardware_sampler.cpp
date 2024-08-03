/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_intel/hardware_sampler.hpp"

#include "plssvm/detail/tracking/gpu_intel/level_zero_device_handle_impl.hpp"  // plssvm::detail::tracking::level_zero_device_handle_impl
#include "plssvm/detail/tracking/gpu_intel/utility.hpp"                        // PLSSVM_LEVEL_ZERO_ERROR_CHECK
#include "plssvm/detail/tracking/hardware_sampler.hpp"                         // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"                      // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/tracking/utility.hpp"                                  // plssvm::detail::tracking::durations_from_reference_time
#include "plssvm/detail/utility.hpp"                                           // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                                    // plssvm::exception, plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                                         // plssvm::target_platform

#include "fmt/chrono.h"          // format std::chrono types
#include "fmt/format.h"          // fmt::format
#include "fmt/ranges.h"          // fmt::join
#include "level_zero/ze_api.h"   // Level Zero runtime functions
#include "level_zero/zes_api.h"  // Level Zero runtime functions

#include <chrono>     // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>    // std::size_t
#include <cstdint>    // std::size_t
#include <cstdint>    // std::int32_t
#include <exception>  // std::exception, std::terminate
#include <ios>        // std::ios_base
#include <iostream>   // std::cerr, std::endl
#include <string>     // std::string
#include <thread>     // std::this_thread
#include <utility>    // std::move
#include <vector>     // std::vector

namespace plssvm::detail::tracking {

gpu_intel_hardware_sampler::gpu_intel_hardware_sampler(const std::size_t device_id, const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } {
    // make sure that zeInit is only called once for all instances
    if (instances_++ == 0) {
        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
        // notify that initialization has been finished
        init_finished_ = true;
    } else {
        // wait until init has been finished!
        while (!init_finished_) { }
    }

    // initialize samples -> can't be done beforehand since the device handle can only be initialized after a call to nvmlInit
    device_ = level_zero_device_handle{ device_id };

    // track the Level Zero API version
    ze_driver_handle_t driver = device_.get_impl().driver;
    ze_api_version_t api_version{};
    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDriverGetApiVersion(driver, &api_version));
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "level_zero_api_version", fmt::format("{}.{}", ZE_MAJOR_VERSION(api_version), ZE_MINOR_VERSION(api_version)) }));
}

gpu_intel_hardware_sampler::~gpu_intel_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->has_sampling_started() && !this->has_sampling_stopped()) {
            this->stop_sampling();
        }
        // the level zero runtime has no dedicated shut down or cleanup function
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        std::terminate();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
}

std::string gpu_intel_hardware_sampler::device_identification() const {
    // get the level zero handle from the device
    ze_device_handle_t device = device_.get_impl().device;
    ze_device_properties_t prop{};
    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zeDeviceGetProperties(device, &prop));
    return fmt::format("gpu_intel_device_{}", prop.deviceId);
}

target_platform gpu_intel_hardware_sampler::sampling_target() const {
    return target_platform::gpu_intel;
}

std::string gpu_intel_hardware_sampler::generate_yaml_string(const std::chrono::steady_clock::time_point start_time_point) const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    return fmt::format("\n"
                       "    sampling_interval: {}\n"
                       "    time_points: [{}]\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}\n"
                       "{}",
                       this->sampling_interval(),
                       fmt::join(durations_from_reference_time(this->time_points(), start_time_point), ", "),
                       general_samples_.generate_yaml_string(),
                       clock_samples_.generate_yaml_string(),
                       power_samples_.generate_yaml_string(),
                       memory_samples_.generate_yaml_string(),
                       temperature_samples_.generate_yaml_string());
}

void gpu_intel_hardware_sampler::sampling_loop() {
    // get the level zero handle from the device
    ze_device_handle_t device = device_.get_impl().device;

    // the different handles
    std::vector<zes_freq_handle_t> frequency_handles{};
    std::vector<zes_pwr_handle_t> power_handles{};
    std::vector<zes_mem_handle_t> memory_handles{};
    std::vector<zes_psu_handle_t> psu_handles{};
    std::vector<zes_temp_handle_t> temperature_handles{};

    //
    // add samples where we only have to retrieve the value once
    //

    this->add_time_point(std::chrono::steady_clock::now());

    // retrieve initial general information
    {
        ze_device_properties_t ze_device_prop{};
        if (zeDeviceGetProperties(device, &ze_device_prop) == ZE_RESULT_SUCCESS) {
            general_samples_.num_threads_per_eu_ = ze_device_prop.numThreadsPerEU;
            general_samples_.eu_simd_width_ = ze_device_prop.physicalEUSimdWidth;
        }

        zes_device_properties_t zes_device_prop{};
        if (zesDeviceGetProperties(device, &zes_device_prop) == ZE_RESULT_SUCCESS) {
            general_samples_.name_ = zes_device_prop.modelName;
        }

        std::uint32_t num_standby_domains{ 0 };
        if (zesDeviceEnumStandbyDomains(device, &num_standby_domains, nullptr) == ZE_RESULT_SUCCESS) {
            std::vector<zes_standby_handle_t> standby_handles(num_standby_domains);
            if (zesDeviceEnumStandbyDomains(device, &num_standby_domains, standby_handles.data()) == ZE_RESULT_SUCCESS) {
                if (!standby_handles.empty()) {
                    // NOTE: only the first standby domain is used here
                    zes_standby_promo_mode_t mode{};
                    if (zesStandbyGetMode(standby_handles.front(), &mode) == ZE_RESULT_SUCCESS) {
                        std::string standby_mode_name{ "unknown" };
                        switch (mode) {
                            case ZES_STANDBY_PROMO_MODE_DEFAULT:
                                standby_mode_name = "default";
                                break;
                            case ZES_STANDBY_PROMO_MODE_NEVER:
                                standby_mode_name = "never";
                                break;
                            default:
                                // do nothing
                                break;
                        }
                        general_samples_.standby_mode_ = std::move(standby_mode_name);
                    }
                }
            }
        }
    }

    // retrieve initial clock related information
    {
        std::uint32_t num_frequency_domains{ 0 };
        if (zesDeviceEnumFrequencyDomains(device, &num_frequency_domains, nullptr) == ZE_RESULT_SUCCESS) {
            frequency_handles.resize(num_frequency_domains);
            if (zesDeviceEnumFrequencyDomains(device, &num_frequency_domains, frequency_handles.data()) == ZE_RESULT_SUCCESS) {
                for (zes_freq_handle_t handle : frequency_handles) {
                    // get frequency properties
                    zes_freq_properties_t prop{};
                    if (zesFrequencyGetProperties(handle, &prop)) {
                        // determine the frequency domain (e.g. GPU, memory, etc)
                        switch (prop.type) {
                            case ZES_FREQ_DOMAIN_GPU:
                                clock_samples_.clock_gpu_min_ = prop.min;
                                clock_samples_.clock_gpu_max_ = prop.max;
                                break;
                            case ZES_FREQ_DOMAIN_MEMORY:
                                clock_samples_.clock_mem_min_ = prop.min;
                                clock_samples_.clock_mem_max_ = prop.max;
                                break;
                            default:
                                // do nothing
                                break;
                        }

                        // get possible frequencies
                        std::uint32_t num_available_clocks{ 0 };
                        if (zesFrequencyGetAvailableClocks(handle, &num_available_clocks, nullptr) == ZE_RESULT_SUCCESS) {
                            std::vector<double> available_clocks(num_available_clocks);
                            if (zesFrequencyGetAvailableClocks(handle, &num_available_clocks, available_clocks.data()) == ZE_RESULT_SUCCESS) {
                                // determine the frequency domain (e.g. GPU, memory, etc)
                                switch (prop.type) {
                                    case ZES_FREQ_DOMAIN_GPU:
                                        clock_samples_.available_clocks_gpu_ = available_clocks;
                                        break;
                                    case ZES_FREQ_DOMAIN_MEMORY:
                                        clock_samples_.available_clocks_mem_ = available_clocks;
                                        break;
                                    default:
                                        // do nothing
                                        break;
                                }
                            }
                        }

                        // get current frequency information
                        zes_freq_state_t frequency_state{};
                        if (zesFrequencyGetState(handle, &frequency_state) == ZE_RESULT_SUCCESS) {
                            // determine the frequency domain (e.g. GPU, memory, etc)
                            switch (prop.type) {
                                case ZES_FREQ_DOMAIN_GPU:
                                    {
                                        if (frequency_state.tdp >= 0.0) {
                                            clock_samples_.tdp_frequency_limit_gpu_ = decltype(clock_samples_.tdp_frequency_limit_gpu_)::value_type{ frequency_state.tdp };
                                        }
                                        if (frequency_state.actual >= 0.0) {
                                            clock_samples_.clock_gpu_ = decltype(clock_samples_.clock_gpu_)::value_type{ frequency_state.actual };
                                        }
                                        if (frequency_state.throttleReasons >= 0.0) {
                                            using vector_type = decltype(clock_samples_.throttle_reason_gpu_)::value_type;
                                            clock_samples_.throttle_reason_gpu_ = vector_type{ static_cast<vector_type::value_type>(frequency_state.throttleReasons) };
                                        }
                                    }
                                    break;
                                case ZES_FREQ_DOMAIN_MEMORY:
                                    {
                                        if (frequency_state.tdp >= 0.0) {
                                            clock_samples_.tdp_frequency_limit_mem_ = decltype(clock_samples_.tdp_frequency_limit_mem_)::value_type{ frequency_state.tdp };
                                        }
                                        if (frequency_state.actual >= 0.0) {
                                            clock_samples_.clock_mem_ = decltype(clock_samples_.clock_mem_)::value_type{ frequency_state.actual };
                                        }
                                        if (frequency_state.throttleReasons >= 0.0) {
                                            using vector_type = decltype(clock_samples_.throttle_reason_mem_)::value_type;
                                            clock_samples_.throttle_reason_mem_ = vector_type{ static_cast<vector_type::value_type>(frequency_state.throttleReasons) };
                                        }
                                    }
                                    break;
                                default:
                                    // do nothing
                                    break;
                            }
                        }
                    }
                }
            }
        }
    }

    // retrieve initial power related information
    {
        std::uint32_t num_power_domains{ 0 };
        if (zesDeviceEnumPowerDomains(device, &num_power_domains, nullptr) == ZE_RESULT_SUCCESS) {
            power_handles.resize(num_power_domains);
            if (zesDeviceEnumPowerDomains(device, &num_power_domains, power_handles.data()) == ZE_RESULT_SUCCESS) {
                if (!power_handles.empty()) {
                    // NOTE: only the first power domain is used here
                    // get total power consumption
                    zes_power_energy_counter_t energy_counter{};
                    if (zesPowerGetEnergyCounter(power_handles.front(), &energy_counter) == ZE_RESULT_SUCCESS) {
                        power_samples_.power_total_energy_consumption_ = decltype(power_samples_.power_total_energy_consumption_)::value_type{ energy_counter.energy };
                    }

                    // get energy thresholds
                    zes_energy_threshold_t energy_threshold{};
                    if (zesPowerGetEnergyThreshold(power_handles.front(), &energy_threshold) == ZE_RESULT_SUCCESS) {
                        power_samples_.energy_threshold_enabled_ = static_cast<decltype(power_samples_.energy_threshold_enabled_)::value_type>(energy_threshold.enable);
                        power_samples_.energy_threshold_ = energy_threshold.threshold;
                    }
                }
            }
        }
    }

    // retrieve initial memory related information
    {
        std::uint32_t num_memory_modules{ 0 };
        if (zesDeviceEnumMemoryModules(device, &num_memory_modules, nullptr) == ZE_RESULT_SUCCESS) {
            memory_handles.resize(num_memory_modules);
            if (zesDeviceEnumMemoryModules(device, &num_memory_modules, memory_handles.data()) == ZE_RESULT_SUCCESS) {
                for (zes_mem_handle_t handle : memory_handles) {
                    zes_mem_properties_t prop{};
                    if (zesMemoryGetProperties(handle, &prop) == ZE_RESULT_SUCCESS) {
                        // get the memory module name
                        const std::string memory_module_name = memory_module_to_name(prop.type);

                        if (prop.physicalSize > 0) {
                            // first value to add -> initialize map
                            if (!memory_samples_.memory_total_.has_value()) {
                                memory_samples_.memory_total_ = decltype(memory_samples_.memory_total_)::value_type{};
                            }
                            // add new physical size
                            memory_samples_.memory_total_.value()[memory_module_name] = prop.physicalSize;
                        }
                        if (prop.busWidth != -1) {
                            // first value to add -> initialize map
                            if (!memory_samples_.bus_width_.has_value()) {
                                memory_samples_.bus_width_ = decltype(memory_samples_.bus_width_)::value_type{};
                            }
                            // add new memory bus width
                            memory_samples_.bus_width_.value()[memory_module_name] = prop.busWidth;
                        }
                        if (prop.numChannels != -1) {
                            // first value to add -> initialize map
                            if (!memory_samples_.num_channels_.has_value()) {
                                memory_samples_.num_channels_ = decltype(memory_samples_.num_channels_)::value_type{};
                            }
                            // add new number of memory channels
                            memory_samples_.num_channels_.value()[memory_module_name] = prop.numChannels;
                        }
                        // first value to add -> initialize map
                        if (!memory_samples_.location_.has_value()) {
                            memory_samples_.location_ = decltype(memory_samples_.location_)::value_type{};
                        }
                        memory_samples_.location_.value()[memory_module_name] = memory_location_to_name(prop.location);

                        // get current memory information
                        zes_mem_state_t mem_state{};
                        if (zesMemoryGetState(handle, &mem_state) == ZE_RESULT_SUCCESS) {
                            // first value to add -> initialize map
                            if (!memory_samples_.allocatable_memory_total_.has_value()) {
                                memory_samples_.allocatable_memory_total_ = decltype(memory_samples_.allocatable_memory_total_)::value_type{};
                            }
                            memory_samples_.allocatable_memory_total_.value()[memory_module_name] = mem_state.size;

                            // first value to add -> initialize map
                            if (!memory_samples_.memory_free_.has_value()) {
                                memory_samples_.memory_free_ = decltype(memory_samples_.memory_free_)::value_type{};
                            }
                            memory_samples_.memory_free_.value()[memory_module_name].push_back(mem_state.free);
                        }
                    }
                }

                // the maximum PCIe stats
                zes_pci_properties_t pci_prop{};
                if (zesDevicePciGetProperties(device, &pci_prop) == ZE_RESULT_SUCCESS) {
                    if (pci_prop.maxSpeed.gen != -1) {
                        memory_samples_.max_pcie_link_generation_ = pci_prop.maxSpeed.gen;
                    }
                    if (pci_prop.maxSpeed.width != -1) {
                        memory_samples_.pcie_max_width_ = pci_prop.maxSpeed.width;
                    }
                    if (pci_prop.maxSpeed.maxBandwidth != -1) {
                        memory_samples_.pcie_link_max_speed_ = pci_prop.maxSpeed.maxBandwidth;
                    }
                }

                // the current PCIe stats
                zes_pci_state_t pci_state{};
                if (zesDevicePciGetState(device, &pci_state) == ZE_RESULT_SUCCESS) {
                    if (pci_state.speed.maxBandwidth != -1) {
                        memory_samples_.pcie_link_speed_ = decltype(memory_samples_.pcie_link_speed_)::value_type{ pci_state.speed.maxBandwidth };
                    }
                    if (pci_state.speed.width != -1) {
                        memory_samples_.pcie_link_width_ = decltype(memory_samples_.pcie_link_width_)::value_type{ pci_state.speed.width };
                    }
                    if (pci_state.speed.gen != -1) {
                        memory_samples_.pcie_link_generation_ = decltype(memory_samples_.pcie_link_generation_)::value_type{ pci_state.speed.gen };
                    }
                }
            }
        }
    }

    // retrieve initial temperature related information
    {
        std::uint32_t num_psus{ 0 };
        if (zesDeviceEnumPsus(device, &num_psus, nullptr) == ZE_RESULT_SUCCESS) {
            psu_handles.resize(num_psus);
            if (zesDeviceEnumPsus(device, &num_psus, psu_handles.data()) == ZE_RESULT_SUCCESS) {
                if (!psu_handles.empty()) {
                    // NOTE: only the first PSU is used here
                    zes_psu_state_t psu_state{};
                    if (zesPsuGetState(psu_handles.front(), &psu_state) == ZE_RESULT_SUCCESS) {
                        if (psu_state.temperature != -1) {
                            temperature_samples_.temperature_psu_ = decltype(temperature_samples_.temperature_psu_)::value_type{ psu_state.temperature };
                        }
                    }
                }
            }
        }

        std::uint32_t num_temperature_sensors{ 0 };
        if (zesDeviceEnumTemperatureSensors(device, &num_temperature_sensors, nullptr) == ZE_RESULT_SUCCESS) {
            temperature_handles.resize(num_temperature_sensors);
            if (zesDeviceEnumTemperatureSensors(device, &num_temperature_sensors, temperature_handles.data()) == ZE_RESULT_SUCCESS) {
                for (zes_temp_handle_t handle : temperature_handles) {
                    zes_temp_properties_t prop{};
                    if (zesTemperatureGetProperties(handle, &prop) == ZE_RESULT_SUCCESS) {
                        const std::string sensor_name = temperature_sensor_type_to_name(prop.type);
                        if (sensor_name.empty()) {
                            // unsupported sensor type
                            continue;
                        }

                        // first value to add -> initialize map
                        if (!temperature_samples_.temperature_max_.has_value()) {
                            temperature_samples_.temperature_max_ = decltype(temperature_samples_.temperature_max_)::value_type{};
                        }
                        // add new maximum temperature
                        temperature_samples_.temperature_max_.value()[sensor_name] = prop.maxTemperature;

                        // first value to add -> initialize map
                        if (!temperature_samples_.temperature_.has_value()) {
                            temperature_samples_.temperature_ = decltype(temperature_samples_.temperature_)::value_type{};
                        }
                        double temp{};
                        if (zesTemperatureGetState(handle, &temp) == ZE_RESULT_SUCCESS) {
                            temperature_samples_.temperature_.value()[sensor_name].push_back(temp);
                        }
                    }
                }
            }
        }
    }

    //
    // loop until stop_sampling() is called
    //

    while (!this->has_sampling_stopped()) {
        // only sample values if the sampler currently isn't paused
        if (this->is_sampling()) {
            // add current time point
            this->add_time_point(std::chrono::steady_clock::now());

            // retrieve clock related samples
            {
                for (zes_freq_handle_t handle : frequency_handles) {
                    // get frequency properties
                    zes_freq_properties_t prop{};
                    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesFrequencyGetProperties(handle, &prop));

                    // get current frequency information
                    zes_freq_state_t frequency_state{};
                    if (clock_samples_.clock_gpu_.has_value() || clock_samples_.clock_mem_.has_value()) {
                        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesFrequencyGetState(handle, &frequency_state));
                        // determine the frequency domain (e.g. GPU, memory, etc)
                        switch (prop.type) {
                            case ZES_FREQ_DOMAIN_GPU:
                                {
                                    if (clock_samples_.tdp_frequency_limit_gpu_.has_value()) {
                                        clock_samples_.tdp_frequency_limit_gpu_->push_back(frequency_state.tdp);
                                    }
                                    if (clock_samples_.clock_gpu_.has_value()) {
                                        clock_samples_.clock_gpu_->push_back(frequency_state.actual);
                                    }
                                    if (clock_samples_.throttle_reason_gpu_.has_value()) {
                                        clock_samples_.throttle_reason_gpu_->push_back(static_cast<decltype(clock_samples_.throttle_reason_gpu_)::value_type::value_type>(frequency_state.throttleReasons));
                                    }
                                }
                                break;
                            case ZES_FREQ_DOMAIN_MEMORY:
                                {
                                    if (clock_samples_.tdp_frequency_limit_mem_.has_value()) {
                                        clock_samples_.tdp_frequency_limit_mem_->push_back(frequency_state.tdp);
                                    }
                                    if (clock_samples_.clock_mem_.has_value()) {
                                        clock_samples_.clock_mem_->push_back(frequency_state.actual);
                                    }
                                    if (clock_samples_.throttle_reason_mem_.has_value()) {
                                        clock_samples_.throttle_reason_mem_->push_back(static_cast<decltype(clock_samples_.throttle_reason_mem_)::value_type::value_type>(frequency_state.throttleReasons));
                                    }
                                }
                                break;
                            default:
                                // do nothing
                                break;
                        }
                    }
                }
            }

            // retrieve power related samples
            {
                if (!power_handles.empty()) {
                    // NOTE: only the first power domain is used here
                    if (power_samples_.power_total_energy_consumption_.has_value()) {
                        // get total power consumption
                        zes_power_energy_counter_t energy_counter{};
                        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesPowerGetEnergyCounter(power_handles.front(), &energy_counter));

                        power_samples_.power_total_energy_consumption_->push_back(energy_counter.energy);
                    }
                }
            }

            // retrieve memory related samples
            {
                for (zes_mem_handle_t handle : memory_handles) {
                    zes_mem_properties_t prop{};
                    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesMemoryGetProperties(handle, &prop));

                    // get the memory module name
                    const std::string memory_module_name = memory_module_to_name(prop.type);

                    if (memory_samples_.memory_free_.has_value()) {
                        // get current memory information
                        zes_mem_state_t mem_state{};
                        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesMemoryGetState(handle, &mem_state));

                        memory_samples_.memory_free_.value()[memory_module_name].push_back(mem_state.free);
                    }
                }

                if (memory_samples_.pcie_link_speed_.has_value() || memory_samples_.pcie_link_width_.has_value() || memory_samples_.pcie_link_width_.has_value()) {
                    // the current PCIe stats
                    zes_pci_state_t pci_state{};
                    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesDevicePciGetState(device, &pci_state));
                    if (memory_samples_.pcie_link_speed_.has_value()) {
                        memory_samples_.pcie_link_speed_->push_back(pci_state.speed.maxBandwidth);
                    }
                    if (memory_samples_.pcie_link_width_.has_value()) {
                        memory_samples_.pcie_link_width_->push_back(pci_state.speed.width);
                    }
                    if (memory_samples_.pcie_link_width_.has_value()) {
                        memory_samples_.pcie_link_generation_->push_back(pci_state.speed.gen);
                    }
                }
            }

            // retrieve temperature related samples
            {
                if (!psu_handles.empty()) {
                    if (temperature_samples_.temperature_psu_.has_value()) {
                        // NOTE: only the first PSU is used here
                        zes_psu_state_t psu_state{};
                        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesPsuGetState(psu_handles.front(), &psu_state));
                        temperature_samples_.temperature_psu_->push_back(psu_state.temperature);
                    }
                }

                for (zes_temp_handle_t handle : temperature_handles) {
                    zes_temp_properties_t prop{};
                    PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesTemperatureGetProperties(handle, &prop));

                    const std::string sensor_name = temperature_sensor_type_to_name(prop.type);
                    if (sensor_name.empty()) {
                        // unsupported sensor type
                        continue;
                    }

                    if (temperature_samples_.temperature_.has_value() && detail::contains(temperature_samples_.temperature_.value(), sensor_name)) {
                        double temp{};
                        PLSSVM_LEVEL_ZERO_ERROR_CHECK(zesTemperatureGetState(handle, &temp));
                        temperature_samples_.temperature_.value()[sensor_name].push_back(temp);
                    }
                }
            }
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

std::ostream &operator<<(std::ostream &out, const gpu_intel_hardware_sampler &sampler) {
    if (sampler.is_sampling()) {
        out.setstate(std::ios_base::failbit);
        return out;
    } else {
        return out << fmt::format("sampling interval: {}\n"
                                  "time points: [{}]\n\n"
                                  "general samples:\n{}\n\n"
                                  "clock samples:\n{}\n\n"
                                  "power samples:\n{}\n\n"
                                  "memory samples:\n{}\n\n"
                                  "temperature samples:\n{}",
                                  sampler.sampling_interval(),
                                  fmt::join(time_points_to_epoch(sampler.time_points()), ", "),
                                  sampler.general_samples(),
                                  sampler.clock_samples(),
                                  sampler.power_samples(),
                                  sampler.memory_samples(),
                                  sampler.temperature_samples());
    }
}

}  // namespace plssvm::detail::tracking
