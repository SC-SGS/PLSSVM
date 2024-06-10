/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/turbostat_hardware_sampler.hpp"

#include "plssvm/detail/assert.hpp"                     // PLSSVM_ASSERT
#include "plssvm/detail/string_conversion.hpp"          // plssvm::detail::split_as
#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/utility.hpp"                    // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"             // plssvm::hardware_sampling_exception

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join
#include "subprocess.h"  // subprocess_p, subprocess_option_e, subprocess_create, subprocess_stdout, subprocess_destroy

#include <array>          // std::array
#include <chrono>         // std::chrono::{steady_clock, duration_cast, milliseconds}
#include <cstddef>        // std::size_t
#include <regex>          // std::regex, std::regex::extended, std::regex_match
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <tuple>          // std::tuple
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::move
#include <vector>         // std::vector

namespace plssvm::detail::tracking {

#if defined(PLSSVM_HARDWARE_SAMPLING_ERROR_CHECKS_ENABLED)

    #define PLSSVM_TURBOSTAT_ERROR_CHECK(turbostat_func)                                   \
        {                                                                                  \
            const int errc = turbostat_func;                                               \
            if (errc != 0) {                                                               \
                throw hardware_sampling_exception{ "Error calling subprocess function!" }; \
            }                                                                              \
        }

#else
    #define PLSSVM_TURBOSTAT_ERROR_CHECK(turbostat_func) turbostat_func;
#endif

turbostat_hardware_sampler::turbostat_hardware_sampler(const std::chrono::milliseconds sampling_interval) :
    hardware_sampler{ sampling_interval } { }

std::string turbostat_hardware_sampler::device_identification() const noexcept {
    return "turbostat_device_cpu";
}

std::string turbostat_hardware_sampler::assemble_yaml_sample_string() const {
    // output the basic sample information
    std::string str = fmt::format("\n"
                                  "    samples:\n"
                                  "      sampling_interval: {}\n",
                                  this->sampling_interval());

    // parse read data in a usable format
    if (data_lines_.empty()) {
        // no samples given (maybe the sample interval was too large)
        return str;
    }

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
    const std::vector<std::string> header = detail::split_as<std::string>(data_lines_.front(), '\t');

    // parse actual samples
    const std::size_t num_samples = data_lines_.size() - 1;  // don't count the column headers
    std::vector<std::vector<double>> data_samples(header.size(), std::vector<double>(num_samples));
#pragma omp parallel for
    for (std::size_t line = 1; line < data_lines_.size(); ++line) {  // skip first line
        // get the values
        const std::vector<double> sample = detail::split_as<double>(data_lines_[line], '\t');

        // check size
        PLSSVM_ASSERT(sample.size() == header.size(), "Invalid data sample size for sample {}!: {} (sample) vs. {} (header)", line - 1, sample.size(), header.size());

        // add current sample to all samples
        for (std::size_t h = 0; h < sample.size(); ++h) {
            data_samples[h][line - 1] = sample[h];
        }
    }

    // calculate time differences
    std::vector<std::chrono::milliseconds> times(num_samples);
    for (std::size_t header_idx = 0; header_idx < header.size(); ++header_idx) {
        if (header[header_idx] == "Time_Of_Day_Seconds") {
            const auto start_epoch = std::chrono::duration<double>{ this->sampling_system_clock_start_time().time_since_epoch() };
            // parse the time information
#pragma omp parallel for
            for (std::size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
                const auto sample_epoch = std::chrono::duration<double>{ data_samples[header_idx][sample_idx] };
                times[sample_idx] = std::chrono::duration_cast<std::chrono::milliseconds>(sample_epoch - start_epoch);
            }
        }
    }
    str += fmt::format("      time_points: [{}]\n",
                       fmt::join(times, ", "));

    // create a map to group the output in categories
    std::unordered_map<std::string, std::vector<std::string>> category_groups{};

    // output data in YAML format
    for (std::size_t header_idx = 0; header_idx < header.size(); ++header_idx) {
        if (header[header_idx] == "Time_Of_Day_Seconds") {
            // already handled
            continue;
        } else if (detail::contains(header_to_name_unit_map, header[header_idx])) {
            const auto &[category, yaml_entry_name, unit] = header_to_name_unit_map.at(header[header_idx]);
            // better category name and unit available
            category_groups[category].push_back(fmt::format("        {}:\n"
                                                            "          turbostat_name: \"{}\"\n"
                                                            "          unit: \"{}\"\n"
                                                            "          values: [{}]\n",
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

                    category_groups[category].push_back(fmt::format("        {}:\n"
                                                                    "          turbostat_name: \"{}\"\n"
                                                                    "          unit: \"{}\"\n"
                                                                    "          values: [{}]\n",
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
                category_groups["general"].push_back(fmt::format("        {}:\n"
                                                                 "          values: [{}]\n",
                                                                 header[header_idx],
                                                                 fmt::join(data_samples[header_idx], ", ")));
            }
        }
    }

    // create final string with grouped entries
    for (const auto &[category, entries] : category_groups) {
        str += fmt::format("      {}:\n"
                           "{}",
                           category,
                           fmt::join(entries, ""));
    }

    // remove last newline
    str.pop_back();

    return str;
}

void turbostat_hardware_sampler::sampling_loop() {
    // -i, --interval    sampling interval in seconds (decimal number)
    // -S, --Summary     limits output to 1-line per interval
    // -q, --quiet       skip decoding system configuration header
    // -e, --enable      enable the additional Time_Of_Day_Seconds column

    const std::string interval = fmt::format("{}", std::chrono::duration<double>{ this->sampling_interval() }.count());
    const int options = subprocess_option_e::subprocess_option_search_user_path | subprocess_option_e::subprocess_option_enable_async;

#if defined(PLSSVM_HARDWARE_TRACKING_VIA_TURBOSTAT_ROOT)
    // must use sudo
    const std::array<const char *, 9> command_line = { "sudo", "turbostat", "-i", interval.data(), "-S", "-q", "-e", "Time_Of_Day_Seconds", nullptr };
#else
    // can run without sudo
    const std::array<const char *, 8> command_line = { "turbostat", "-i", interval.data(), "-S", "-q", "-e", "Time_Of_Day_Seconds", nullptr };
#endif

    // create subprocess
    subprocess_s proc{};
    PLSSVM_TURBOSTAT_ERROR_CHECK(subprocess_create(command_line.data(), options, &proc));

    //
    // loop until stop_sampling() is called
    //

    std::string buffer(static_cast<std::string::size_type>(4096), '\0');  // 4096 character should be enough
    while (!sampling_stopped_) {
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
                    data_lines_.push_back(std::move(sample));
                }
            }
        }
    }

    // terminate subprocess -> same as strg + c for turbostat
    PLSSVM_TURBOSTAT_ERROR_CHECK(subprocess_destroy(&proc));
}

}  // namespace plssvm::detail::tracking
