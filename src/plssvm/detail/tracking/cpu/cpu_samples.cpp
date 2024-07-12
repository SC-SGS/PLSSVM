/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/cpu/cpu_samples.hpp"

#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::value_or_default

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <optional>     // std::optional
#include <ostream>      // std::ostream
#include <regex>        // std::regex, std::regex_match
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string cpu_general_samples::generate_yaml_string() const {
    std::string str{ "    general:\n" };

    // architecture
    if (this->architecture_.has_value()) {
        str += fmt::format("      architecture:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->architecture_.value());
    }
    // byte order
    if (this->byte_order_.has_value()) {
        str += fmt::format("      byte_order:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->byte_order_.value());
    }
    // number of threads including hyper-threads
    if (this->num_threads_.has_value()) {
        str += fmt::format("      num_threads:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_threads_.value());
    }
    // number of threads per core
    if (this->threads_per_core_.has_value()) {
        str += fmt::format("      threads_per_core:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->threads_per_core_.value());
    }
    // number of cores per socket
    if (this->cores_per_socket_.has_value()) {
        str += fmt::format("      cores_per_socket:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->cores_per_socket_.value());
    }
    // number of cores per socket
    if (this->num_sockets_.has_value()) {
        str += fmt::format("      num_sockets:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->num_sockets_.value());
    }
    // number of NUMA nodes
    if (this->numa_nodes_.has_value()) {
        str += fmt::format("      numa_nodes:\n"
                           "        unit: \"int\"\n"
                           "        values: {}\n",
                           this->numa_nodes_.value());
    }
    // the vendor specific ID
    if (this->vendor_id_.has_value()) {
        str += fmt::format("      vendor_id:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->vendor_id_.value());
    }
    // the CPU name
    if (this->name_.has_value()) {
        str += fmt::format("      name:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->name_.value());
    }
    // CPU specific flags (like SSE, AVX, ...)
    if (this->flags_.has_value()) {
        str += fmt::format("      flags:\n"
                           "        unit: \"string\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->flags_.value(), ", "));
    }

    // the percent the CPU was busy
    if (this->busy_percent_.has_value()) {
        str += fmt::format("      utilization:\n"
                           "        turbostat_name: \"Busy%\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->busy_percent_.value(), ", "));
    }
    // the instructions per cycle count
    if (this->ipc_.has_value()) {
        str += fmt::format("      instructions_per_cycle:\n"
                           "        turbostat_name: \"IPC\"\n"
                           "        unit: \"float\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->ipc_.value(), ", "));
    }
    // the number of interrupts
    if (this->irq_.has_value()) {
        str += fmt::format("      interrupts:\n"
                           "        turbostat_name: \"IRQ\"\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->irq_.value(), ", "));
    }
    // the number of system management interrupts
    if (this->smi_.has_value()) {
        str += fmt::format("      system_management_interrupts:\n"
                           "        turbostat_name: \"SMI\"\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->smi_.value(), ", "));
    }
    // the number of times the CPU was in the poll state
    if (this->poll_.has_value()) {
        str += fmt::format("      polling_state:\n"
                           "        turbostat_name: \"POLL\"\n"
                           "        unit: \"int\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->poll_.value(), ", "));
    }
    // the percent the CPU was in the polling state
    if (this->poll_percent_.has_value()) {
        str += fmt::format("      polling_percentage:\n"
                           "        turbostat_name: \"POLL%\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->poll_percent_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_general_samples &samples) {
    std::string str = fmt::format("architecture: {}\n"
                                  "byte_order: {}\n"
                                  "num_threads: {}\n"
                                  "threads_per_core: {}\n"
                                  "cores_per_socket: {}\n"
                                  "num_sockets: {}\n"
                                  "numa_nodes: {}\n"
                                  "vendor_id: {}\n"
                                  "name: {}\n"
                                  "flags: [{}]\n"
                                  "busy_percent: [{}]\n"
                                  "ipc: [{}]\n"
                                  "irq: [{}]\n"
                                  "smi: [{}]\n"
                                  "poll: [{}]\n"
                                  "poll_percent: [{}]",
                                  value_or_default(samples.get_architecture()),
                                  value_or_default(samples.get_byte_order()),
                                  value_or_default(samples.get_num_threads()),
                                  value_or_default(samples.get_threads_per_core()),
                                  value_or_default(samples.get_cores_per_socket()),
                                  value_or_default(samples.get_num_sockets()),
                                  value_or_default(samples.get_numa_nodes()),
                                  value_or_default(samples.get_vendor_id()),
                                  value_or_default(samples.get_name()),
                                  fmt::join(value_or_default(samples.get_flags()), ", "),
                                  fmt::join(value_or_default(samples.get_busy_percent()), ", "),
                                  fmt::join(value_or_default(samples.get_ipc()), ", "),
                                  fmt::join(value_or_default(samples.get_irq()), ", "),
                                  fmt::join(value_or_default(samples.get_smi()), ", "),
                                  fmt::join(value_or_default(samples.get_poll()), ", "),
                                  fmt::join(value_or_default(samples.get_poll_percent()), ", "));

    // remove last newline
    str.pop_back();

    return out << str;
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string cpu_clock_samples::generate_yaml_string() const {
    std::string str{ "    clock:\n" };

    // true if frequency boost is enabled
    if (this->frequency_boost_.has_value()) {
        str += fmt::format("      frequency_boost:\n"
                           "        unit: \"bool\"\n"
                           "        values: {}\n",
                           this->frequency_boost_.value());
    }
    // the minimal CPU frequency
    if (this->min_frequency_.has_value()) {
        str += fmt::format("      min_cpu_frequency:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->min_frequency_.value());
    }
    // the maximum CPU frequency
    if (this->max_frequency_.has_value()) {
        str += fmt::format("      max_cpu_frequency:\n"
                           "        unit: \"MHz\"\n"
                           "        values: {}\n",
                           this->max_frequency_.value());
    }

    // the average CPU frequency
    if (this->average_frequency_.has_value()) {
        str += fmt::format("      average_frequency:\n"
                           "        turbostat_name: \"Avg_MHz\"\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->average_frequency_.value(), ", "));
    }
    // the average CPU frequency excluding idle time
    if (this->average_non_idle_frequency_.has_value()) {
        str += fmt::format("      average_non_idle_frequency:\n"
                           "        turbostat_name: \"Bzy_MHz\"\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->average_non_idle_frequency_.value(), ", "));
    }
    // the time stamp counter
    if (this->time_stamp_counter_.has_value()) {
        str += fmt::format("      time_stamp_counter:\n"
                           "        turbostat_name: \"TSC_MHz\"\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->time_stamp_counter_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_clock_samples &samples) {
    return out << fmt::format("frequency_boost: {}\n"
                              "min_frequency: {}\n"
                              "max_frequency: {}\n"
                              "average_frequency: [{}]\n"
                              "average_non_idle_frequency: [{}]\n"
                              "time_stamp_counter: [{}]",
                              value_or_default(samples.get_frequency_boost()),
                              value_or_default(samples.get_min_frequency()),
                              value_or_default(samples.get_max_frequency()),
                              fmt::join(value_or_default(samples.get_average_frequency()), ", "),
                              fmt::join(value_or_default(samples.get_average_non_idle_frequency()), ", "),
                              fmt::join(value_or_default(samples.get_time_stamp_counter()), ", "));
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string cpu_power_samples::generate_yaml_string() const {
    std::string str{ "    power:\n" };

    // the package Watt
    if (this->package_watt_.has_value()) {
        str += fmt::format("      package_power:\n"
                           "        turbostat_name: \"PkgWatt\"\n"
                           "        unit: \"W\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->package_watt_.value(), ", "));
    }
    // the core Watt
    if (this->core_watt_.has_value()) {
        str += fmt::format("      core_power:\n"
                           "        turbostat_name: \"CorWatt\"\n"
                           "        unit: \"W\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->core_watt_.value(), ", "));
    }
    // the DRAM Watt
    if (this->ram_watt_.has_value()) {
        str += fmt::format("      dram_power:\n"
                           "        turbostat_name: \"RAMWatt\"\n"
                           "        unit: \"W\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->ram_watt_.value(), ", "));
    }
    // the percent of time when the RAPL package throttle was active
    if (this->package_rapl_throttle_percent_.has_value()) {
        str += fmt::format("      package_rapl_throttling:\n"
                           "        turbostat_name: \"PKG_%\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->package_rapl_throttle_percent_.value(), ", "));
    }
    // the percent of time when the RAPL DRAM throttle was active
    if (this->dram_rapl_throttle_percent_.has_value()) {
        str += fmt::format("      dram_rapl_throttling:\n"
                           "        turbostat_name: \"RAM_%\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->dram_rapl_throttle_percent_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_power_samples &samples) {
    return out << fmt::format("package_watt: [{}]\n"
                              "core_watt: [{}]\n"
                              "ram_watt: [{}]\n"
                              "package_rapl_throttle_percent: [{}]\n"
                              "dram_rapl_throttle_percent: [{}]",
                              fmt::join(value_or_default(samples.get_package_watt()), ", "),
                              fmt::join(value_or_default(samples.get_core_watt()), ", "),
                              fmt::join(value_or_default(samples.get_ram_watt()), ", "),
                              fmt::join(value_or_default(samples.get_package_rapl_throttle_percent()), ", "),
                              fmt::join(value_or_default(samples.get_dram_rapl_throttle_percent()), ", "));
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string cpu_memory_samples::generate_yaml_string() const {
    std::string str{ "    memory:\n" };

    // the size of the L1 data cache
    if (this->l1d_cache_.has_value()) {
        str += fmt::format("      cache_size_L1d:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->l1d_cache_.value());
    }
    // the size of the L1 instruction cache
    if (this->l1i_cache_.has_value()) {
        str += fmt::format("      cache_size_L1i:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->l1i_cache_.value());
    }
    // the size of the L2 cache
    if (this->l2_cache_.has_value()) {
        str += fmt::format("      cache_size_L2:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->l2_cache_.value());
    }
    // the size of the L3 cache
    if (this->l3_cache_.has_value()) {
        str += fmt::format("      cache_size_L3:\n"
                           "        unit: \"string\"\n"
                           "        values: \"{}\"\n",
                           this->l3_cache_.value());
    }

    // the total size of available memory
    if (this->memory_total_.has_value()) {
        str += fmt::format("      memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->memory_total_.value());
    }
    // the total size of the swap memory
    if (this->swap_memory_total_.has_value()) {
        str += fmt::format("      swap_memory_total:\n"
                           "        unit: \"B\"\n"
                           "        values: {}\n",
                           this->swap_memory_total_.value());
    }

    // the available free memory
    if (this->memory_free_.has_value()) {
        str += fmt::format("      memory_free:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_free_.value(), ", "));
    }
    // the used memory
    if (this->memory_used_.has_value()) {
        str += fmt::format("      memory_used:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->memory_used_.value(), ", "));
    }
    // the available swap memory
    if (this->swap_memory_free_.has_value()) {
        str += fmt::format("      swap_memory_free:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->swap_memory_free_.value(), ", "));
    }
    // the swap memory
    if (this->swap_memory_used_.has_value()) {
        str += fmt::format("      swap_memory_used:\n"
                           "        unit: \"B\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->swap_memory_used_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_memory_samples &samples) {
    return out << fmt::format("l1d_cache: {}\n"
                              "l1i_cache: {}\n"
                              "l2_cache: {}\n"
                              "l3_cache: {}\n"
                              "memory_total: {}\n"
                              "swap_memory_total: {}\n"
                              "memory_free: [{}]\n"
                              "memory_used: [{}]\n"
                              "swap_memory_free: [{}]\n"
                              "swap_memory_used: [{}]",
                              value_or_default(samples.get_l1d_cache()),
                              value_or_default(samples.get_l1i_cache()),
                              value_or_default(samples.get_l2_cache()),
                              value_or_default(samples.get_l3_cache()),
                              value_or_default(samples.get_memory_total()),
                              value_or_default(samples.get_swap_memory_total()),
                              fmt::join(value_or_default(samples.get_memory_free()), ", "),
                              fmt::join(value_or_default(samples.get_memory_used()), ", "),
                              fmt::join(value_or_default(samples.get_swap_memory_free()), ", "),
                              fmt::join(value_or_default(samples.get_swap_memory_used()), ", "));
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string cpu_temperature_samples::generate_yaml_string() const {
    std::string str{ "    temperature:\n" };

    // the temperature of the cores
    if (this->core_temperature_.has_value()) {
        str += fmt::format("      per_core_temperature:\n"
                           "        turbostat_name: \"CoreTmp\"\n"
                           "        unit: \"°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->core_temperature_.value(), ", "));
    }
    // the percentage of time the core throttled due the temperature constraints
    if (this->core_throttle_percent_.has_value()) {
        str += fmt::format("      core_throttle_percentage:\n"
                           "        turbostat_name: \"CoreThr\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->core_throttle_percent_.value(), ", "));
    }
    // the temperature of the whole package
    if (this->package_temperature_.has_value()) {
        str += fmt::format("      per_package_temperature:\n"
                           "        turbostat_name: \"PkgTmp\"\n"
                           "        unit: \"°C\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->package_temperature_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_temperature_samples &samples) {
    return out << fmt::format("core_temperature: [{}]\n"
                              "core_throttle_percent: [{}]\n"
                              "package_temperature: [{}]",
                              fmt::join(value_or_default(samples.get_core_temperature()), ", "),
                              fmt::join(value_or_default(samples.get_core_throttle_percent()), ", "),
                              fmt::join(value_or_default(samples.get_package_temperature()), ", "));
}

//*************************************************************************************************************************************//
//                                                          gfx (iGPU) samples                                                         //
//*************************************************************************************************************************************//

std::string cpu_gfx_samples::generate_yaml_string() const {
    std::string str{ "    integrated_gpu:\n" };

    // the percentage of time the iGPU was in the render state
    if (this->gfx_render_state_percent_.has_value()) {
        str += fmt::format("      graphics_render_state:\n"
                           "        turbostat_name: \"GFX%rc6\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->gfx_render_state_percent_.value(), ", "));
    }
    // the core frequency of the iGPU
    if (this->gfx_frequency_.has_value()) {
        str += fmt::format("      graphics_frequency:\n"
                           "        turbostat_name: \"GFXMHz\"\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->gfx_frequency_.value(), ", "));
    }
    // the average core frequency of the iGPU
    if (this->average_gfx_frequency_.has_value()) {
        str += fmt::format("      average_graphics_frequency:\n"
                           "        turbostat_name: \"GFXAMHz\"\n"
                           "        unit: \"MHz\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->average_gfx_frequency_.value(), ", "));
    }
    // the percentage of time the iGPU was in the c0 state
    if (this->gfx_state_c0_percent_.has_value()) {
        str += fmt::format("      gpu_state_c0:\n"
                           "        turbostat_name: \"GFX%C0\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->gfx_state_c0_percent_.value(), ", "));
    }
    // the percentage of time the CPU worked for the iGPU
    if (this->cpu_works_for_gpu_percent_.has_value()) {
        str += fmt::format("      cpu_works_for_gpu:\n"
                           "        turbostat_name: \"CPUGFX%\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->cpu_works_for_gpu_percent_.value(), ", "));
    }
    // the iGPU Watt
    if (this->gfx_watt_.has_value()) {
        str += fmt::format("      graphics_power:\n"
                           "        turbostat_name: \"GFXWatt\"\n"
                           "        unit: \"W\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->gfx_watt_.value(), ", "));
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_gfx_samples &samples) {
    return out << fmt::format("gfx_render_state_percent: [{}]\n"
                              "gfx_frequency: [{}]\n"
                              "average_gfx_frequency: [{}]\n"
                              "gfx_state_c0_percent: [{}]\n"
                              "cpu_works_for_gpu_percent: [{}]\n"
                              "gfx_watt: [{}]",
                              fmt::join(value_or_default(samples.get_gfx_render_state_percent()), ", "),
                              fmt::join(value_or_default(samples.get_gfx_frequency()), ", "),
                              fmt::join(value_or_default(samples.get_average_gfx_frequency()), ", "),
                              fmt::join(value_or_default(samples.get_gfx_state_c0_percent()), ", "),
                              fmt::join(value_or_default(samples.get_cpu_works_for_gpu_percent()), ", "),
                              fmt::join(value_or_default(samples.get_gfx_watt()), ", "));
}

//*************************************************************************************************************************************//
//                                                          idle state samples                                                         //
//*************************************************************************************************************************************//

std::string cpu_idle_states_samples::generate_yaml_string() const {
    std::string str{ "    idle_states:\n" };

    // the percentage of time all CPUs were in the c0 state
    if (this->all_cpus_state_c0_percent_.has_value()) {
        str += fmt::format("      all_cpus_state_c0:\n"
                           "        turbostat_name: \"Totl%C0\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->all_cpus_state_c0_percent_.value(), ", "));
    }
    // the percentage of time any CPU was in the c0 state
    if (this->any_cpu_state_c0_percent_.has_value()) {
        str += fmt::format("      any_cpu_state_c0:\n"
                           "        turbostat_name: \"Any%C0\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->any_cpu_state_c0_percent_.value(), ", "));
    }
    // the percentage of time the CPUs were in the low power idle state
    if (this->low_power_idle_state_percent_.has_value()) {
        str += fmt::format("      lower_power_idle_state:\n"
                           "        turbostat_name: \"CPU%LPI\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->low_power_idle_state_percent_.value(), ", "));
    }
    // the percentage of time the CPUs were in the system low power idle state
    if (this->system_low_power_idle_state_percent_.has_value()) {
        str += fmt::format("      system_lower_power_idle_state:\n"
                           "        turbostat_name: \"SYS%LPI\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->system_low_power_idle_state_percent_.value(), ", "));
    }
    // the percentage of time the package was in the low power idle state
    if (this->package_low_power_idle_state_percent_.has_value()) {
        str += fmt::format("      package_lower_power_idle_state:\n"
                           "        turbostat_name: \"Pkg%LPI\"\n"
                           "        unit: \"percentage\"\n"
                           "        values: [{}]\n",
                           fmt::join(this->package_low_power_idle_state_percent_.value(), ", "));
    }

    // the other core idle states
    if (this->idle_states_.has_value()) {
        constexpr static std::array<std::tuple<std::string_view, std::string_view, std::string_view, std::size_t, std::size_t>, 5> turbostat_regex{
            std::tuple{ "CPU%[0-9a-zA-Z]+", "cpu_idle_state_{}_percentage", "percentage", 4, 0 },
            std::tuple{ "Pkg%[0-9a-zA-Z]+", "package_idle_state_{}_percentage", "percentage", 4, 0 },
            std::tuple{ "Pk%[0-9a-zA-Z]+", "package_idle_state_{}_percentage", "percentage", 3, 0 },
            std::tuple{ "C[0-9a-zA-Z]+%", "idle_state_{}_percentage", "percentage", 1, 1 },
            std::tuple{ "C[0-9a-zA-Z]+", "idle_state_{}", "int", 1, 0 }
        };

        for (const auto &[entry, values] : this->idle_states_.value()) {
            // test all regular expressions one after another
            for (const auto &[regex, entry_name_placeholder, entry_unit, prefix_size, suffix_size] : turbostat_regex) {
                const std::regex reg{ std::string{ regex }, std::regex::extended };

                if (std::regex_match(entry, reg)) {
                    // remove specified prefix and suffix from state
                    std::string_view state{ entry };
                    state.remove_prefix(prefix_size);
                    state.remove_suffix(suffix_size);

                    // assemble better category name
                    const std::string entry_name_with_state = fmt::format(fmt::runtime(entry_name_placeholder), state);

                    str += fmt::format("      {}:\n"
                                       "        turbostat_name: \"{}\"\n"
                                       "        unit: \"{}\"\n"
                                       "        values: [{}]\n",
                                       entry_name_with_state,
                                       entry,
                                       entry_unit,
                                       fmt::join(values, ", "));
                    break;
                }
            }
        }
    }

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const cpu_idle_states_samples &samples) {
    std::string str = fmt::format("all_cpus_state_c0_percent: [{}]\n"
                                  "any_cpu_state_c0_percent: [{}]\n"
                                  "low_power_idle_state_percent: [{}]\n"
                                  "system_low_power_idle_state_percent: [{}]\n"
                                  "package_low_power_idle_state_percent: [{}]\n",
                                  fmt::join(value_or_default(samples.get_all_cpus_state_c0_percent()), ", "),
                                  fmt::join(value_or_default(samples.get_any_cpu_state_c0_percent()), ", "),
                                  fmt::join(value_or_default(samples.get_low_power_idle_state_percent()), ", "),
                                  fmt::join(value_or_default(samples.get_system_low_power_idle_state_percent()), ", "),
                                  fmt::join(value_or_default(samples.get_package_low_power_idle_state_percent()), ", "));

    // add map entries
    if (samples.get_idle_states().has_value()) {
        for (const auto &[key, value] : samples.get_idle_states().value()) {
            str += fmt::format("{}: [{}]\n", key, fmt::join(value, ", "));
        }
    }

    // remove last newline
    str.pop_back();

    return out << str;
}

}  // namespace plssvm::detail::tracking
