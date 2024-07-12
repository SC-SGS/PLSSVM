/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"

#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/string_conversion.hpp"             // plssvm::detail::split_as
#include "plssvm/detail/string_utility.hpp"                // plssvm::detail::{starts_with, trim}
#include "plssvm/detail/tracking/cpu/cpu_samples.hpp"      // plssvm::detail::tracking::{cpu_general_samples, clock_samples, power_samples, memory_samples, temperature_samples, gfx_samples, idle_state_samples}
#include "plssvm/detail/tracking/cpu/utility.hpp"          // PLSSVM_SUBPROCESS_ERROR_CHECK
#include "plssvm/detail/tracking/hardware_sampler.hpp"     // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/tracking/utility.hpp"              // plssvm::detail::tracking::{durations_from_reference_time, time_points_to_epoch}
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::hardware_sampling_exception
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <chrono>         // std::chrono::{steady_clock, milliseconds}
#include <cstddef>        // std::size_t
#include <cstdio>         // std::FILE, std::fread
#include <exception>      // std::exception, std::terminate
#include <ios>            // std::ios_base
#include <iostream>       // std::cerr << std::endl
#include <optional>       // std::make_optional
#include <ostream>        // std::ostream
#include <regex>          // std::regex, std::regex::extended, std::regex_match, std::regex_replace
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <thread>         // std::this_thread
#include <unordered_map>  // std::unordered_map
#include <vector>         // std::vector

namespace plssvm::detail::tracking {

cpu_hardware_sampler::cpu_hardware_sampler(const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } {
    // track the lscpu version
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    {
        const std::string lscpu_output = run_subprocess("lscpu --version");
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "lscpu_version", detail::trim(lscpu_output) }));
    }
#endif

    // track the free version
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_FREE_ENABLED)
    {
        const std::string free_output = run_subprocess("free --version");
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "free_version", detail::trim(free_output) }));
    }
#endif

    // track the turbostat version
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
    {
        const std::string turbostat_output = run_subprocess("turbostat --version");
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "turbostat_version", detail::trim(turbostat_output) }));
    }
#endif
}

cpu_hardware_sampler::~cpu_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->has_sampling_started() && !this->has_sampling_stopped()) {
            this->stop_sampling();
        }
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        std::terminate();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::terminate();
    }
}

std::string cpu_hardware_sampler::device_identification() const {
    return "cpu_device";
}

target_platform cpu_hardware_sampler::sampling_target() const {
    return target_platform::cpu;
}

std::string cpu_hardware_sampler::generate_yaml_string([[maybe_unused]] const std::chrono::steady_clock::time_point start_time_point) const {
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
                       "{}\n"
                       "{}\n"
                       "{}",
                       this->sampling_interval(),
                       fmt::join(durations_from_reference_time(this->time_points(), start_time_point), ", "),
                       general_samples_.generate_yaml_string(),
                       clock_samples_.generate_yaml_string(),
                       power_samples_.generate_yaml_string(),
                       memory_samples_.generate_yaml_string(),
                       temperature_samples_.generate_yaml_string(),
                       gfx_samples_.generate_yaml_string(),
                       idle_state_samples_.generate_yaml_string());
}

void cpu_hardware_sampler::sampling_loop() {
    //
    // add samples where we only have to retrieve the value once
    //

    this->add_time_point(std::chrono::steady_clock::now());

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    {
        const std::string lscpu_output = run_subprocess("lscpu");
        const std::vector<std::string_view> lscpu_lines = detail::split(detail::trim(lscpu_output), '\n');

        for (std::string_view line : lscpu_lines) {
            line = detail::trim(line);
            // extract the value
            std::string_view value{ line };
            value.remove_prefix(value.find_first_of(":") + 1);
            value = detail::trim(value);

            // check the lines if the start with an entry that we want to sample
            if (detail::starts_with(line, "Architecture")) {
                general_samples_.architecture_ = detail::convert_to<decltype(general_samples_.architecture_)::value_type>(value);
            } else if (detail::starts_with(line, "Byte Order")) {
                general_samples_.byte_order_ = detail::convert_to<decltype(general_samples_.byte_order_)::value_type>(value);
            } else if (detail::starts_with(line, "CPU(s)")) {
                general_samples_.num_threads_ = detail::convert_to<decltype(general_samples_.num_threads_)::value_type>(value);
            } else if (detail::starts_with(line, "Thread(s) per core")) {
                general_samples_.threads_per_core_ = detail::convert_to<decltype(general_samples_.threads_per_core_)::value_type>(value);
            } else if (detail::starts_with(line, "Core(s) per socket")) {
                general_samples_.cores_per_socket_ = detail::convert_to<decltype(general_samples_.cores_per_socket_)::value_type>(value);
            } else if (detail::starts_with(line, "Socket(s)")) {
                general_samples_.num_sockets_ = detail::convert_to<decltype(general_samples_.num_sockets_)::value_type>(value);
            } else if (detail::starts_with(line, "NUMA node(s)")) {
                general_samples_.numa_nodes_ = detail::convert_to<decltype(general_samples_.numa_nodes_)::value_type>(value);
            } else if (detail::starts_with(line, "Vendor ID")) {
                general_samples_.vendor_id_ = detail::convert_to<decltype(general_samples_.vendor_id_)::value_type>(value);
            } else if (detail::starts_with(line, "Model name")) {
                general_samples_.name_ = detail::convert_to<decltype(general_samples_.name_)::value_type>(value);
            } else if (detail::starts_with(line, "Flags")) {
                general_samples_.flags_ = detail::split_as<decltype(general_samples_.flags_)::value_type::value_type>(value, ' ');
            } else if (detail::starts_with(line, "Frequency boost")) {
                clock_samples_.frequency_boost_ = value == "enabled";
            } else if (detail::starts_with(line, "CPU max MHz")) {
                clock_samples_.min_frequency_ = detail::convert_to<decltype(clock_samples_.min_frequency_)::value_type>(value);
            } else if (detail::starts_with(line, "CPU min MHz")) {
                clock_samples_.max_frequency_ = detail::convert_to<decltype(clock_samples_.max_frequency_)::value_type>(value);
            } else if (detail::starts_with(line, "L1d cache")) {
                memory_samples_.l1d_cache_ = detail::convert_to<decltype(memory_samples_.l1d_cache_)::value_type>(value);
            } else if (detail::starts_with(line, "L1i cache")) {
                memory_samples_.l1i_cache_ = detail::convert_to<decltype(memory_samples_.l1i_cache_)::value_type>(value);
            } else if (detail::starts_with(line, "L2 cache")) {
                memory_samples_.l2_cache_ = detail::convert_to<decltype(memory_samples_.l2_cache_)::value_type>(value);
            } else if (detail::starts_with(line, "L3 cache")) {
                memory_samples_.l3_cache_ = detail::convert_to<decltype(memory_samples_.l3_cache_)::value_type>(value);
            }
        }
    }
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_FREE_ENABLED)
    const std::regex whitespace_replace_reg{ "[ ]+", std::regex::extended };
    {
        std::string free_output = run_subprocess("free -b");
        free_output = std::regex_replace(free_output, whitespace_replace_reg, " ");
        const std::vector<std::string_view> free_lines = detail::split(detail::trim(free_output), '\n');
        PLSSVM_ASSERT(free_lines.size() >= 3, "Must read exactly more than three lines, but only read {} lines!", free_lines.size());

        // read memory information
        const std::vector<std::string_view> memory_data = detail::split(free_lines[1], ' ');
        memory_samples_.memory_total_ = detail::convert_to<decltype(memory_samples_.memory_total_)::value_type>(memory_data[1]);
        memory_samples_.memory_used_ = decltype(memory_samples_.memory_used_)::value_type{ detail::convert_to<decltype(memory_samples_.memory_used_)::value_type::value_type>(memory_data[2]) };
        memory_samples_.memory_free_ = decltype(memory_samples_.memory_free_)::value_type{ detail::convert_to<decltype(memory_samples_.memory_free_)::value_type::value_type>(memory_data[3]) };

        // read swap information
        const std::vector<std::string_view> swap_data = detail::split(free_lines[2], ' ');
        memory_samples_.swap_memory_total_ = detail::convert_to<decltype(memory_samples_.swap_memory_total_)::value_type>(swap_data[1]);
        memory_samples_.swap_memory_used_ = decltype(memory_samples_.swap_memory_used_)::value_type{ detail::convert_to<decltype(memory_samples_.swap_memory_used_)::value_type::value_type>(swap_data[2]) };
        memory_samples_.swap_memory_free_ = decltype(memory_samples_.swap_memory_free_)::value_type{ detail::convert_to<decltype(memory_samples_.swap_memory_free_)::value_type::value_type>(swap_data[3]) };
    }
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)

    // -n, --num_iterations     number of the measurement iterations
    // -i, --interval           sampling interval in seconds (decimal number)
    // -S, --Summary            limits output to 1-line per interval
    // -q, --quiet              skip decoding system configuration header

    // get header information
    #if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ROOT)
    // run with sudo
    const std::string_view turbostat_command_line = "sudo turbostat -n 1 -i 0.001 -S -q";
    #else
    // run without sudo
    const std::string_view turbostat_command_line = "turbostat -n 1 -i 0.001 -S -q";
    #endif

    {
        // run turbostat
        const std::string turbostat_output = run_subprocess(turbostat_command_line);

        // retrieve the turbostat data
        const std::vector<std::string_view> data = detail::split(detail::trim(turbostat_output), '\n');
        PLSSVM_ASSERT(data.size() == 2, "Must read exactly two lines, but read {} lines!", data.size());
        const std::vector<std::string_view> header = detail::split(data[0], '\t');
        const std::vector<std::string_view> values = detail::split(data[1], '\t');

        for (std::size_t i = 0; i < header.size(); ++i) {
            if (header[i] == "Avg_MHz") {
                using vector_type = decltype(clock_samples_.average_frequency_)::value_type;
                clock_samples_.average_frequency_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "Busy%") {
                using vector_type = decltype(general_samples_.busy_percent_)::value_type;
                general_samples_.busy_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "Bzy_MHz") {
                using vector_type = decltype(clock_samples_.average_non_idle_frequency_)::value_type;
                clock_samples_.average_non_idle_frequency_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "TSC_MHz") {
                using vector_type = decltype(clock_samples_.time_stamp_counter_)::value_type;
                clock_samples_.time_stamp_counter_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "IPC") {
                using vector_type = decltype(general_samples_.ipc_)::value_type;
                general_samples_.ipc_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "IRQ") {
                using vector_type = decltype(general_samples_.irq_)::value_type;
                general_samples_.irq_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "SMI") {
                using vector_type = decltype(general_samples_.smi_)::value_type;
                general_samples_.smi_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "POLL") {
                using vector_type = decltype(general_samples_.poll_)::value_type;
                general_samples_.poll_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "POLL%") {
                using vector_type = decltype(general_samples_.poll_percent_)::value_type;
                general_samples_.poll_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "CoreTmp") {
                using vector_type = decltype(temperature_samples_.core_temperature_)::value_type;
                temperature_samples_.core_temperature_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "CoreThr") {
                using vector_type = decltype(temperature_samples_.core_throttle_percent_)::value_type;
                temperature_samples_.core_throttle_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "PkgTmp") {
                using vector_type = decltype(temperature_samples_.package_temperature_)::value_type;
                temperature_samples_.package_temperature_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "GFX%rc6") {
                using vector_type = decltype(gfx_samples_.gfx_render_state_percent_)::value_type;
                gfx_samples_.gfx_render_state_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "GFXMHz") {
                using vector_type = decltype(gfx_samples_.gfx_frequency_)::value_type;
                gfx_samples_.gfx_frequency_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "GFXAMHz") {
                using vector_type = decltype(gfx_samples_.average_gfx_frequency_)::value_type;
                gfx_samples_.average_gfx_frequency_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "Totl%C0") {
                using vector_type = decltype(idle_state_samples_.all_cpus_state_c0_percent_)::value_type;
                idle_state_samples_.all_cpus_state_c0_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "Any%C0") {
                using vector_type = decltype(idle_state_samples_.any_cpu_state_c0_percent_)::value_type;
                idle_state_samples_.any_cpu_state_c0_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "GFX%C0") {
                using vector_type = decltype(gfx_samples_.gfx_state_c0_percent_)::value_type;
                gfx_samples_.gfx_state_c0_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "CPUGFX%") {
                using vector_type = decltype(gfx_samples_.cpu_works_for_gpu_percent_)::value_type;
                gfx_samples_.cpu_works_for_gpu_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "CPU%LPI") {
                using vector_type = decltype(idle_state_samples_.low_power_idle_state_percent_)::value_type;
                idle_state_samples_.low_power_idle_state_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "SYS%LPI") {
                using vector_type = decltype(idle_state_samples_.system_low_power_idle_state_percent_)::value_type;
                idle_state_samples_.system_low_power_idle_state_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "Pkg%LPI") {
                using vector_type = decltype(idle_state_samples_.package_low_power_idle_state_percent_)::value_type;
                idle_state_samples_.package_low_power_idle_state_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "PkgWatt") {
                using vector_type = decltype(power_samples_.package_watt_)::value_type;
                idle_state_samples_.package_low_power_idle_state_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "CorWatt") {
                using vector_type = decltype(power_samples_.core_watt_)::value_type;
                power_samples_.core_watt_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "GFXWatt") {
                using vector_type = decltype(gfx_samples_.gfx_watt_)::value_type;
                gfx_samples_.gfx_watt_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "RAMWatt") {
                using vector_type = decltype(power_samples_.ram_watt_)::value_type;
                power_samples_.ram_watt_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "PKG_%") {
                using vector_type = decltype(power_samples_.package_rapl_throttle_percent_)::value_type;
                power_samples_.package_rapl_throttle_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else if (header[i] == "RAM_%") {
                using vector_type = decltype(power_samples_.dram_rapl_throttle_percent_)::value_type;
                power_samples_.dram_rapl_throttle_percent_ = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
            } else {
                // test against regex
                const std::string header_str{ header[i] };
                const std::regex reg{ std::string{ "CPU%[0-9a-zA-Z]+|Pkg%[0-9a-zA-Z]+|Pk%[0-9a-zA-Z]+|C[0-9a-zA-Z]+%|C[0-9a-zA-Z]+" }, std::regex::extended };
                if (std::regex_match(header_str, reg)) {
                    // first time this branch is reached -> create optional value
                    if (!idle_state_samples_.idle_states_.has_value()) {
                        idle_state_samples_.idle_states_ = std::make_optional<typename cpu_idle_states_samples::map_type>();
                    }

                    using vector_type = cpu_idle_states_samples::map_type::mapped_type;
                    idle_state_samples_.idle_states_.value()[header_str] = vector_type{ detail::convert_to<typename vector_type::value_type>(values[i]) };
                }
            }
        }
    }
#endif

    //
    // loop until stop_sampling() is called
    //

    while (!this->has_sampling_stopped()) {
        // only sample values if the sampler currently isn't paused
        if (this->is_sampling()) {
            // add current time point
            this->add_time_point(std::chrono::steady_clock::now());

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_FREE_ENABLED)
            {
                // run free
                std::string free_output = run_subprocess("free -b");
                free_output = std::regex_replace(free_output, whitespace_replace_reg, " ");
                const std::vector<std::string_view> free_lines = detail::split(detail::trim(free_output), '\n');
                PLSSVM_ASSERT(free_lines.size() >= 3, "Must read exactly more than three lines, but only read {} lines!", free_lines.size());

                // read memory information
                const std::vector<std::string_view> memory_data = detail::split(free_lines[1], ' ');
                memory_samples_.memory_used_->push_back(detail::convert_to<decltype(memory_samples_.memory_used_)::value_type::value_type>(memory_data[2]));
                memory_samples_.memory_free_->push_back(detail::convert_to<decltype(memory_samples_.memory_free_)::value_type::value_type>(memory_data[3]));

                // read swap information
                const std::vector<std::string_view> swap_data = detail::split(free_lines[2], ' ');
                memory_samples_.swap_memory_used_->push_back(detail::convert_to<decltype(memory_samples_.swap_memory_used_)::value_type::value_type>(swap_data[2]));
                memory_samples_.swap_memory_free_->push_back(detail::convert_to<decltype(memory_samples_.swap_memory_free_)::value_type::value_type>(swap_data[3]));
            }
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
            {
                // run turbostat
                const std::string turbostat_output = run_subprocess(turbostat_command_line);

                // retrieve the turbostat data
                const std::vector<std::string_view> data = detail::split(detail::trim(turbostat_output), '\n');
                PLSSVM_ASSERT(data.size() == 2, "Must read exactly two lines, but read {} lines!", data.size());
                const std::vector<std::string_view> header = detail::split(data[0], '\t');
                const std::vector<std::string_view> values = detail::split(data[1], '\t');

                // add values to the respective sample entries
                for (std::size_t i = 0; i < header.size(); ++i) {
                    if (header[i] == "Avg_MHz") {
                        using vector_type = decltype(clock_samples_.average_frequency_)::value_type;
                        clock_samples_.average_frequency_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "Busy%") {
                        using vector_type = decltype(general_samples_.busy_percent_)::value_type;
                        general_samples_.busy_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "Bzy_MHz") {
                        using vector_type = decltype(clock_samples_.average_non_idle_frequency_)::value_type;
                        clock_samples_.average_non_idle_frequency_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "TSC_MHz") {
                        using vector_type = decltype(clock_samples_.time_stamp_counter_)::value_type;
                        clock_samples_.time_stamp_counter_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "IPC") {
                        using vector_type = decltype(general_samples_.ipc_)::value_type;
                        general_samples_.ipc_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "IRQ") {
                        using vector_type = decltype(general_samples_.irq_)::value_type;
                        general_samples_.irq_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "SMI") {
                        using vector_type = decltype(general_samples_.smi_)::value_type;
                        general_samples_.smi_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "POLL") {
                        using vector_type = decltype(general_samples_.poll_)::value_type;
                        general_samples_.poll_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "POLL%") {
                        using vector_type = decltype(general_samples_.poll_percent_)::value_type;
                        general_samples_.poll_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "CoreTmp") {
                        using vector_type = decltype(temperature_samples_.core_temperature_)::value_type;
                        temperature_samples_.core_temperature_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "CoreThr") {
                        using vector_type = decltype(temperature_samples_.core_throttle_percent_)::value_type;
                        temperature_samples_.core_throttle_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "PkgTmp") {
                        using vector_type = decltype(temperature_samples_.package_temperature_)::value_type;
                        temperature_samples_.package_temperature_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "GFX%rc6") {
                        using vector_type = decltype(gfx_samples_.gfx_render_state_percent_)::value_type;
                        gfx_samples_.gfx_render_state_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "GFXMHz") {
                        using vector_type = decltype(gfx_samples_.gfx_frequency_)::value_type;
                        gfx_samples_.gfx_frequency_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "GFXAMHz") {
                        using vector_type = decltype(gfx_samples_.average_gfx_frequency_)::value_type;
                        gfx_samples_.average_gfx_frequency_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "Totl%C0") {
                        using vector_type = decltype(idle_state_samples_.all_cpus_state_c0_percent_)::value_type;
                        idle_state_samples_.all_cpus_state_c0_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "Any%C0") {
                        using vector_type = decltype(idle_state_samples_.any_cpu_state_c0_percent_)::value_type;
                        idle_state_samples_.any_cpu_state_c0_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "GFX%C0") {
                        using vector_type = decltype(gfx_samples_.gfx_state_c0_percent_)::value_type;
                        gfx_samples_.gfx_state_c0_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "CPUGFX%") {
                        using vector_type = decltype(gfx_samples_.cpu_works_for_gpu_percent_)::value_type;
                        gfx_samples_.cpu_works_for_gpu_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "CPU%LPI") {
                        using vector_type = decltype(idle_state_samples_.low_power_idle_state_percent_)::value_type;
                        idle_state_samples_.low_power_idle_state_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "SYS%LPI") {
                        using vector_type = decltype(idle_state_samples_.system_low_power_idle_state_percent_)::value_type;
                        idle_state_samples_.system_low_power_idle_state_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "Pkg%LPI") {
                        using vector_type = decltype(idle_state_samples_.package_low_power_idle_state_percent_)::value_type;
                        idle_state_samples_.package_low_power_idle_state_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "PkgWatt") {
                        using vector_type = decltype(power_samples_.package_watt_)::value_type;
                        idle_state_samples_.package_low_power_idle_state_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "CorWatt") {
                        using vector_type = decltype(power_samples_.core_watt_)::value_type;
                        power_samples_.core_watt_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "GFXWatt") {
                        using vector_type = decltype(gfx_samples_.gfx_watt_)::value_type;
                        gfx_samples_.gfx_watt_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "RAMWatt") {
                        using vector_type = decltype(power_samples_.ram_watt_)::value_type;
                        power_samples_.ram_watt_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "PKG_%") {
                        using vector_type = decltype(power_samples_.package_rapl_throttle_percent_)::value_type;
                        power_samples_.package_rapl_throttle_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else if (header[i] == "RAM_%") {
                        using vector_type = decltype(power_samples_.dram_rapl_throttle_percent_)::value_type;
                        power_samples_.dram_rapl_throttle_percent_->push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                    } else {
                        const std::string header_str{ header[i] };
                        if (detail::contains(idle_state_samples_.idle_states_.value(), header_str)) {
                            using vector_type = cpu_idle_states_samples::map_type::mapped_type;
                            idle_state_samples_.idle_states_.value()[header_str].push_back(detail::convert_to<typename vector_type::value_type>(values[i]));
                        }
                    }
                }
            }
#endif
        }

        // wait for the sampling interval to pass to retrieve the next sample
        std::this_thread::sleep_for(this->sampling_interval());
    }
}

std::ostream &operator<<(std::ostream &out, const cpu_hardware_sampler &sampler) {
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
                                  "temperature samples:\n{}\n\n"
                                  "gfx samples:\n{}\n\n"
                                  "idle state samples:\n{}",
                                  sampler.sampling_interval(),
                                  fmt::join(time_points_to_epoch(sampler.time_points()), ", "),
                                  sampler.general_samples(),
                                  sampler.clock_samples(),
                                  sampler.power_samples(),
                                  sampler.memory_samples(),
                                  sampler.temperature_samples(),
                                  sampler.gfx_samples(),
                                  sampler.idle_state_samples());
    }
}

}  // namespace plssvm::detail::tracking
