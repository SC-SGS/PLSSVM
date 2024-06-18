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
#include "plssvm/detail/tracking/hardware_sampler.hpp"     // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/detail/tracking/utility.hpp"              // plssvm::detail::tracking::durations_from_reference_time
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::hardware_sampling_exception

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join
#include "subprocess.h"  // subprocess_p, subprocess_option_e, subprocess_create, subprocess_stdout, subprocess_destroy

#include <array>          // std::array
#include <chrono>         // std::chrono::{system_clock, milliseconds}
#include <cstddef>        // std::size_t
#include <cstdio>         // std::FILE, std::fread
#include <exception>      // std::exception, std::terminate
#include <iostream>       // std::cerr << std::endl
#include <regex>          // std::regex, std::regex::extended, std::regex_match
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <tuple>          // std::tuple
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::move
#include <vector>         // std::vector

namespace plssvm::detail::tracking {

#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)

    #define PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_func)                                                                      \
        {                                                                                                                       \
            const int errc = subprocess_func;                                                                                   \
            if (errc != 0) {                                                                                                    \
                throw hardware_sampling_exception{ fmt::format("Error calling subprocess function \"{}\"", #subprocess_func) }; \
            }                                                                                                                   \
        }

#else
    #define PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_func) subprocess_func;
#endif

cpu_hardware_sampler::cpu_hardware_sampler(const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } {
    const int options = subprocess_option_e::subprocess_option_search_user_path | subprocess_option_e::subprocess_option_enable_async;

    // track the lscpu version
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    {
        const std::array<const char *, 3> command_line = { "lscpu", "--version", nullptr };

        // create subprocess
        subprocess_s proc{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_create(command_line.data(), options, &proc));
        // wait until process has finished
        int return_code{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_join(&proc, &return_code));
        if (return_code != 0) {
            throw hardware_sampling_exception{ fmt::format("Error: lscpu returned with {}!", return_code) };
        }
        // get stdout handle and read data
        std::FILE *stdout_handle = subprocess_stdout(&proc);
        std::string buffer(static_cast<std::string::size_type>(512), '\0');  // 512 characters should be enough
        const std::size_t bytes_read = std::fread(buffer.data(), sizeof(typename decltype(buffer)::value_type), buffer.size(), stdout_handle);
        if (bytes_read == 0) {
            throw hardware_sampling_exception{ "Error in lscpu: no bytes were read!" };
        }
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "lscpu_version", buffer.substr(0, buffer.find_first_of('\n')) }));
    }
#endif

    // track the turbostat version
#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
    {
        const std::array<const char *, 3> command_line = { "turbostat", "--version", nullptr };

        // create subprocess
        subprocess_s proc{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_create(command_line.data(), options, &proc));
        // wait until process has finished
        int return_code{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_join(&proc, &return_code));
        if (return_code != 0) {
            throw hardware_sampling_exception{ fmt::format("Error: lscpu returned with {}!", return_code) };
        }
        // get stdout handle and read data
        std::FILE *stdout_handle = subprocess_stderr(&proc);
        std::string buffer(static_cast<std::string::size_type>(512), '\0');  // 512 characters should be enough
        const std::size_t bytes_read = std::fread(buffer.data(), sizeof(typename decltype(buffer)::value_type), buffer.size(), stdout_handle);
        if (bytes_read == 0) {
            throw hardware_sampling_exception{ "Error in lscpu: no bytes were read!" };
        }
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "dependencies", "turbostat_version", buffer.substr(0, buffer.find_first_of('\n')) }));
    }
#endif
}

cpu_hardware_sampler::~cpu_hardware_sampler() {
    try {
        // if this hardware sampler is still sampling, stop it
        if (this->is_sampling()) {
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

std::string cpu_hardware_sampler::generate_yaml_string([[maybe_unused]] const std::chrono::system_clock::time_point start_time_point) const {
    // check whether it's safe to generate the YAML entry
    if (this->is_sampling()) {
        throw hardware_sampling_exception{ "Can't create the final YAML entry if the hardware sampler is still running!" };
    }

    // output the basic sample information
    std::string str = fmt::format("\n"
                                  "    sampling_interval: {}\n",
                                  this->sampling_interval());

    // create a map to group the output in categories
    std::unordered_map<std::string, std::vector<std::string>> category_groups{};

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    // parse general lscpu information
    if (!lscpu_data_lines_.empty()) {
        // create a lscpu entry mapping including the used units
        static const std::unordered_map<std::string, std::tuple<std::string, std::string, std::string>> entry_to_name_unit_map{
            { "Architecture", { "general", "architecture", "string" } },
            { "Byte Order", { "general", "byte_order", "string" } },
            { "CPU(s)", { "general", "num_threads", "int" } },
            { "Thread(s) per core", { "general", "threads_per_core", "int" } },
            { "Core(s) per core", { "general", "cores_per_socket", "int" } },
            { "NUMA node(s)", { "general", "numa_nodes", "int" } },
            { "Vendor ID", { "general", "vendor_id", "string" } },
            { "Model name", { "general", "cpu_name", "string" } },
            { "Frequency boost", { "clock", "frequency_boost", "bool" } },
            { "CPU max MHz", { "clock", "max_cpu_frequency", "MHz" } },
            { "CPU min MHz", { "clock", "min_cpu_frequency", "MHz" } },
            { "Flags", { "general", "flags", "string" } },
            { "L1d cache", { "memory", "cache_L1d", "string" } },
            { "L1i cache", { "memory", "cache_L1i", "string" } },
            { "L2 cache", { "memory", "cache_L2", "string" } },
            { "L3 cache", { "memory", "cache_L3", "string" } }
        };

        // parse lscpu output
        for (const std::string &line : lscpu_data_lines_) {
            const std::string_view line_view{ detail::trim(line) };

            // check if the line should be used
            for (const auto &[entry_name, values] : entry_to_name_unit_map) {
                if (detail::starts_with(line_view, entry_name)) {
                    const auto &[category, yaml_entry_name, unit] = values;

                    // extract the value
                    std::string_view line_value{ line_view };
                    line_value.remove_prefix(line_value.find_first_of(":") + 1);
                    line_value = detail::trim(line_value);

                    // add entry to category_groups map
                    if (yaml_entry_name == "flags") {
                        // flags should be parsed as an array
                        std::vector<std::string> flags = detail::split_as<std::string>(line_value, ' ');
                        // quote all flags
                        for (std::string &flag : flags) {
                            flag = fmt::format("\"{}\"", flag);
                        }
                        category_groups[category].push_back(fmt::format("      {}:\n"
                                                                        "        unit: \"string\"\n"
                                                                        "        values: [{}]\n",
                                                                        yaml_entry_name,
                                                                        fmt::join(flags, ", ")));
                    } else if (unit == "string") {
                        // strings should be quoted
                        category_groups[category].push_back(fmt::format("      {}:\n"
                                                                        "        unit: \"string\"\n"
                                                                        "        values: \"{}\"\n",
                                                                        yaml_entry_name,
                                                                        line_value));
                    } else if (unit == "bool") {
                        // enabled should be converted to a boolean value
                        const bool enabled = line_value == "enabled";
                        category_groups[category].push_back(fmt::format("      {}:\n"
                                                                        "        unit: \"bool\"\n"
                                                                        "        values: {}\n",
                                                                        yaml_entry_name,
                                                                        enabled));
                    } else {
                        category_groups[category].push_back(fmt::format("      {}:\n"
                                                                        "        unit: \"{}\"\n"
                                                                        "        values: {}\n",
                                                                        yaml_entry_name,
                                                                        unit,
                                                                        line_value));
                    }
                }
            }
        }
    }
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
    // parse read data in a usable format
    if (!turbostat_data_lines_.empty()) {
        // create a more useful turbostat header name mapping including the used units
        static const std::unordered_map<std::string, std::tuple<std::string, std::string, std::string>> header_to_name_unit_map{
            { "Avg_MHz", { "clock", "average_frequency", "MHz" } },
            { "Busy%", { "general", "utilization", "percentage" } },
            { "Bzy_MHz", { "clock", "average_non_idle_frequency", "MHz" } },
            { "TSC_MHz", { "clock", "time_stamp_counter", "MHz" } },
            { "IPC", { "general", "instructions_per_cycle", "float" } },
            { "IRQ", { "general", "interrupts", "int" } },
            { "SMI", { "general", "system_management_interrupts", "int" } },
            { "POLL", { "general", "polling_state", "int" } },
            // use std::regex for C.+
            { "POLL%", { "general", "polling_percentage", "percentage" } },
            // use std::regex for C.+%
            // use std::regex for CPU%.+
            { "CoreTmp", { "temperature", "per_core_temperature", "°C" } },
            { "CoreThr", { "temperature", "core_throttle_percentage", "percentage" } },
            { "PkgTmp", { "temperature", "per_package_temperature", "°C" } },
            { "GFX%rc6", { "integrated_gpu", "graphics_render_state", "percentage" } },
            { "GFXMHz", { "integrated_gpu", "graphics_frequency", "MHz" } },
            { "GFXAMHz", { "integrated_gpu", "average_graphics_frequency", "MHz" } },
            { "Totl%C0", { "idle_states", "all_cpus_state_c0", "percentage" } },
            { "Any%C0", { "idle_states", "any_cpu_state_c0", "percentage" } },
            { "GFX%C0", { "integrated_gpu", "gpu_state_c0", "percentage" } },
            { "CPUGFX%", { "integrated_gpu", "cpu_works_for_gpu", "percentage" } },
            // use std::regex for Pkg%.+
            { "CPU%LPI", { "idle_state", "lower_power_idle_state", "percentage" } },
            { "SYS%LPI", { "idle_state", "system_lower_power_idle_state", "percentage" } },
            { "Pkg%LPI", { "idle_state", "package_lower_power_idle_state", "percentage" } },
            { "PkgWatt", { "power", "package_power", "W" } },
            { "CorWatt", { "power", "core_power", "W" } },
            { "GFXWatt", { "power", "graphics_power", "W" } },
            { "RAMWatt", { "power", "dram_power", "W" } },
            { "PKG_%", { "power", "package_rapl_throttling", "percentage" } },
            { "RAM_%", { "power", "dram_rapl_throttling", "percentage" } }
        };
        // create a more useful turbostat header name mapping including the used units using regular expressions
        static const std::vector<std::tuple<std::string, std::string, std::string, std::string, std::size_t, std::size_t>> regex_header_to_name_unit_map{
            { "CPU%[0-9a-zA-Z]+", "idle_state", "cpu_idle_state_{}_percentage", "percentage", 4, 0 },
            { "Pkg%[0-9a-zA-Z]+", "idle_state", "package_idle_state_{}_percentage", "percentage", 4, 0 },
            { "Pk%[0-9a-zA-Z]+", "idle_state", "package_idle_state_{}_percentage", "percentage", 3, 0 },
            { "C[0-9a-zA-Z]+%", "idle_state", "idle_state_{}_percentage", "percentage", 1, 1 },
            { "C[0-9a-zA-Z]+", "idle_state", "idle_state_{}", "int", 1, 0 }
        };

        // header information
        const std::vector<std::string> header = detail::split_as<std::string>(turbostat_data_lines_.front(), '\t');

        // parse actual samples
        const std::size_t num_samples = turbostat_data_lines_.size() - 1;  // don't count the column headers
        std::vector<std::vector<double>> data_samples(header.size(), std::vector<double>(num_samples));
    #pragma omp parallel for
        for (std::size_t line = 1; line < turbostat_data_lines_.size(); ++line) {  // skip first line
            // get the values
            const std::vector<double> sample = detail::split_as<double>(turbostat_data_lines_[line], '\t');

            // check size
            PLSSVM_ASSERT(sample.size() == header.size(), "Invalid data sample size for sample {}!: {} (sample) vs. {} (header)", line - 1, sample.size(), header.size());

            // add current sample to all samples
            for (std::size_t h = 0; h < sample.size(); ++h) {
                data_samples[h][line - 1] = sample[h];
            }
        }

        // calculate time differences

        for (std::size_t header_idx = 0; header_idx < header.size(); ++header_idx) {
            if (header[header_idx] == "Time_Of_Day_Seconds") {
                std::vector<std::chrono::system_clock::time_point> times(num_samples);
                // parse the time information
    #pragma omp parallel for
                for (std::size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
                    const auto dur = std::chrono::duration<double>{ data_samples[header_idx][sample_idx] };
                    times[sample_idx] = std::chrono::system_clock::time_point{ std::chrono::duration_cast<std::chrono::milliseconds>(dur) };
                }

                str += fmt::format("    time_points: [{}]\n",
                                   fmt::join(durations_from_reference_time(times, start_time_point), ", "));
                break;
            }
        }

        // output data in YAML format
        for (std::size_t header_idx = 0; header_idx < header.size(); ++header_idx) {
            if (header[header_idx] == "Time_Of_Day_Seconds") {
                // already handled
                continue;
            } else if (detail::contains(header_to_name_unit_map, header[header_idx])) {
                const auto &[category, yaml_entry_name, unit] = header_to_name_unit_map.at(header[header_idx]);
                // better category name and unit available
                category_groups[category].push_back(fmt::format("      {}:\n"
                                                                "        turbostat_name: \"{}\"\n"
                                                                "        unit: \"{}\"\n"
                                                                "        values: [{}]\n",
                                                                yaml_entry_name,
                                                                header[header_idx],
                                                                unit,
                                                                fmt::join(data_samples[header_idx], ", ")));
            } else {
                // try using regular expressions for a better category name and unit
                bool regex_found{ false };
                // test all regular expressions one after another
                for (const auto &[regex_value, category, yaml_entry_name, unit, prefix_size, suffix_size] : regex_header_to_name_unit_map) {
                    const std::regex reg{ regex_value, std::regex::extended };

                    // check if regex matches
                    if (std::regex_match(header[header_idx], reg)) {
                        // remove specified prefix and suffix from state
                        std::string_view state{ header[header_idx] };
                        state.remove_prefix(prefix_size);
                        state.remove_suffix(suffix_size);

                        // assemble better category name
                        const std::string yaml_category_str = fmt::format(fmt::runtime(yaml_entry_name), state);

                        category_groups[category].push_back(fmt::format("      {}:\n"
                                                                        "        turbostat_name: \"{}\"\n"
                                                                        "        unit: \"{}\"\n"
                                                                        "        values: [{}]\n",
                                                                        yaml_category_str,
                                                                        header[header_idx],
                                                                        unit,
                                                                        fmt::join(data_samples[header_idx], ", ")));

                        // regex match found
                        regex_found = true;
                        break;
                    }
                }

                // fallback to most simple output if nothing else worked
                if (!regex_found) {
                    category_groups["general"].push_back(fmt::format("      {}:\n"
                                                                     "        values: [{}]\n",
                                                                     header[header_idx],
                                                                     fmt::join(data_samples[header_idx], ", ")));
                }
            }
        }
    }
#endif

    // create final string with grouped entries
    for (const auto &[category, entries] : category_groups) {
        str += fmt::format("    {}:\n"
                           "{}",
                           category,
                           fmt::join(entries, ""));
    }

    // remove last newline
    str.pop_back();

    return str;
}

void cpu_hardware_sampler::sampling_loop() {
    const int options = subprocess_option_e::subprocess_option_search_user_path | subprocess_option_e::subprocess_option_enable_async;

    // TODO: getter + fill time_points_

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_LSCPU_ENABLED)
    {
        const std::array<const char *, 2> command_line = { "lscpu", nullptr };

        // create subprocess
        subprocess_s proc{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_create(command_line.data(), options, &proc));
        // wait until process has finished
        int return_code{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_join(&proc, &return_code));
        if (return_code != 0) {
            throw hardware_sampling_exception{ fmt::format("Error: lscpu returned with {}!", return_code) };
        }
        // get stdout handle and read data
        std::FILE *stdout_handle = subprocess_stdout(&proc);
        std::string buffer(static_cast<std::string::size_type>(4096), '\0');  // 4096 character should be enough
        const std::size_t bytes_read = std::fread(buffer.data(), sizeof(typename decltype(buffer)::value_type), buffer.size(), stdout_handle);
        if (bytes_read == 0) {
            throw hardware_sampling_exception{ "Error in lscpu: no bytes were read!" };
        }
        lscpu_data_lines_ = detail::split_as<std::string>(buffer, '\n');
    }
#endif

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ENABLED)
    {
        // -i, --interval    sampling interval in seconds (decimal number)
        // -S, --Summary     limits output to 1-line per interval
        // -q, --quiet       skip decoding system configuration header
        // -e, --enable      enable the additional Time_Of_Day_Seconds column

        const std::string interval = fmt::format("{}", std::chrono::duration<double>{ this->sampling_interval() }.count());

    #if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ROOT)
        // must use sudo
        const std::array<const char *, 9> command_line = { "sudo", "turbostat", "-i", interval.data(), "-S", "-q", "-e", "Time_Of_Day_Seconds", nullptr };
    #else
        // can run without sudo
        const std::array<const char *, 8> command_line = { "turbostat", "-i", interval.data(), "-S", "-q", "-e", "Time_Of_Day_Seconds", nullptr };
    #endif

        // create subprocess
        subprocess_s proc{};
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_create(command_line.data(), options, &proc));

        //
        // loop until stop_sampling() is called
        //

        std::string buffer(static_cast<std::string::size_type>(4096), '\0');  // 4096 character should be enough
        while (!this->has_sampling_stopped()) {
            // only sample values if the sampler currently isn't paused
            if (this->is_sampling()) {
                // read new stdout line
                const unsigned read_size = subprocess_read_stdout(&proc, buffer.data(), buffer.size());
                // add data if anything was read
                if (read_size > 0u) {
                    // get the read substring
                    const std::string_view read_samples{ buffer.data(), static_cast<std::string::size_type>(read_size) };
                    // split the substring on newlines
                    std::vector<std::string> samples = detail::split_as<std::string>(read_samples, '\n');
                    // append the new lines to the already read lines
                    for (std::string &sample : samples) {
                        if (!sample.empty()) {
                            turbostat_data_lines_.push_back(std::move(sample));
                        }
                    }
                }
            }
        }

        // terminate subprocess -> same as strg + c for turbostat
        PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_destroy(&proc));
    }
#endif
}

}  // namespace plssvm::detail::tracking
