/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/cpu/cpu_samples.hpp"       // plssvm::detail::tracking::{cpu_general_samples, clock_samples, power_samples, memory_samples, temperature_samples, gfx_samples, idle_state_samples}
#include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"  // plssvm::detail::tracking::cpu_hardware_sampler
#include "plssvm/detail/tracking/hardware_sampler.hpp"      // plssvm::detail::tracking::hardware_sampler

#include "fmt/core.h"           // fmt::format
#include "pybind11/chrono.h"    // automatic bindings for std::chrono::milliseconds
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // automatic bindings for std::optional and std::vector

#include <chrono>  // std::chrono::milliseconds

namespace py = pybind11;

void init_cpu_hardware_sampler(py::module_ &m) {
    // use a detail.tracking submodule for the hardware sampling
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    const py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");

    // bind the general samples
    py::class_<plssvm::detail::tracking::cpu_general_samples>(tracking_module, "CpuGeneralSamples")
        .def("get_architecture", &plssvm::detail::tracking::cpu_general_samples::get_architecture, "the CPU architecture (e.g., x86_64)")
        .def("get_byte_order", &plssvm::detail::tracking::cpu_general_samples::get_byte_order, "the byte order (e.g., little/big endian)")
        .def("get_num_threads", &plssvm::detail::tracking::cpu_general_samples::get_num_threads, "the number of threads of the CPU(s) including potential hyper-threads")
        .def("get_threads_per_core", &plssvm::detail::tracking::cpu_general_samples::get_threads_per_core, "the number of hyper-threads per core")
        .def("get_cores_per_socket", &plssvm::detail::tracking::cpu_general_samples::get_cores_per_socket, "the number of physical cores per socket")
        .def("get_num_sockets", &plssvm::detail::tracking::cpu_general_samples::get_num_sockets, "the number of sockets")
        .def("get_numa_nodes", &plssvm::detail::tracking::cpu_general_samples::get_numa_nodes, "the number of NUMA nodes")
        .def("get_vendor_id", &plssvm::detail::tracking::cpu_general_samples::get_vendor_id, "the vendor ID (e.g. GenuineIntel)")
        .def("get_name", &plssvm::detail::tracking::cpu_general_samples::get_name, "the name of the CPU")
        .def("get_flags", &plssvm::detail::tracking::cpu_general_samples::get_flags, "potential CPU flags (e.g., sse4_1, avx, avx, etc)")
        .def("get_busy_percent", &plssvm::detail::tracking::cpu_general_samples::get_busy_percent, "the percent the CPU was busy doing work")
        .def("get_ipc", &plssvm::detail::tracking::cpu_general_samples::get_ipc, "the instructions-per-cycle count")
        .def("get_irq", &plssvm::detail::tracking::cpu_general_samples::get_irq, "the number of interrupts")
        .def("get_smi", &plssvm::detail::tracking::cpu_general_samples::get_smi, "the number of system management interrupts")
        .def("get_poll", &plssvm::detail::tracking::cpu_general_samples::get_poll, "the number of times the CPU was in the polling state")
        .def("get_poll_percent", &plssvm::detail::tracking::cpu_general_samples::get_poll_percent, "the percent of the CPU was in the polling state")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::cpu_clock_samples>(tracking_module, "CpuClockSamples")
        .def("get_frequency_boost", &plssvm::detail::tracking::cpu_clock_samples::get_frequency_boost, "true if frequency boosting is enabled")
        .def("get_min_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_min_frequency, "the minimum possible CPU frequency in MHz")
        .def("get_max_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_max_frequency, "the maximum possible CPU frequency in MHz")
        .def("get_average_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_average_frequency, "the average CPU frequency in MHz including idle cores")
        .def("get_average_non_idle_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_average_non_idle_frequency, "the average CPU frequency in MHz excluding idle cores")
        .def("get_time_stamp_counter", &plssvm::detail::tracking::cpu_clock_samples::get_time_stamp_counter, "the time stamp counter")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::cpu_power_samples>(tracking_module, "CpuPowerSamples")
        .def("get_package_watt", &plssvm::detail::tracking::cpu_power_samples::get_package_watt, "the currently consumed power of the package of the CPU in W")
        .def("get_core_watt", &plssvm::detail::tracking::cpu_power_samples::get_core_watt, "the currently consumed power of the core part of the CPU in W")
        .def("get_ram_watt", &plssvm::detail::tracking::cpu_power_samples::get_ram_watt, "the currently consumed power of the RAM part of the CPU in W")
        .def("get_package_rapl_throttle_percent", &plssvm::detail::tracking::cpu_power_samples::get_package_rapl_throttle_percent, "the percent of time the package throttled due to RAPL limiters")
        .def("get_dram_rapl_throttle_percent", &plssvm::detail::tracking::cpu_power_samples::get_dram_rapl_throttle_percent, "the percent of time the DRAM throttled due to RAPL limiters")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::cpu_memory_samples>(tracking_module, "CpuMemorySamples")
        .def("get_l1d_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l1d_cache, "the size of the L1 data cache")
        .def("get_l1i_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l1i_cache, "the size of the L1 instruction cache")
        .def("get_l2_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l2_cache, "the size of the L2 cache")
        .def("get_l3_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l3_cache, "the size of the L2 cache")
        .def("get_memory_total", &plssvm::detail::tracking::cpu_memory_samples::get_memory_total, "the total available memory in Byte")
        .def("get_swap_memory_total", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_total, "the total available swap memory in Byte")
        .def("get_memory_used", &plssvm::detail::tracking::cpu_memory_samples::get_memory_used, "the currently used memory in Byte")
        .def("get_memory_free", &plssvm::detail::tracking::cpu_memory_samples::get_memory_free, "the currently free memory in Byte")
        .def("get_swap_memory_used", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_used, "the currently used swap memory in Byte")
        .def("get_swap_memory_free", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_free, "the currently free swap memory in Byte")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::cpu_temperature_samples>(tracking_module, "CpuTemperatureSamples")
        .def("get_core_temperature", &plssvm::detail::tracking::cpu_temperature_samples::get_core_temperature, "the current temperature of the core part of the CPU in °C")
        .def("get_core_throttle_percent", &plssvm::detail::tracking::cpu_temperature_samples::get_core_throttle_percent, "the percent of time the CPU has throttled")
        .def("get_package_temperature", &plssvm::detail::tracking::cpu_temperature_samples::get_package_temperature, "the current temperature of the whole package in °C")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_temperature_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuTemperatureSamples with\n{}\n>", self);
        });

    // bind the gfx samples
    py::class_<plssvm::detail::tracking::cpu_gfx_samples>(tracking_module, "CpuGfxSamples")
        .def("get_gfx_render_state_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_render_state_percent, "the percent of time the iGPU was in the render state")
        .def("get_gfx_frequency", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_frequency, "the current iGPU power consumption in W")
        .def("get_average_gfx_frequency", &plssvm::detail::tracking::cpu_gfx_samples::get_average_gfx_frequency, "the average iGPU frequency in MHz")
        .def("get_gfx_state_c0_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_state_c0_percent, "the percent of the time the iGPU was in the c0 state")
        .def("get_cpu_works_for_gpu_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_cpu_works_for_gpu_percent, "the percent of time the CPU was doing work for the iGPU")
        .def("get_gfx_watt", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_watt, "the currently consumed power of the iGPU of the CPU in W")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_gfx_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuGfxSamples with\n{}\n>", self);
        });

    // bind the idle state samples
    py::class_<plssvm::detail::tracking::cpu_idle_states_samples>(tracking_module, "CpuIdleStateSamples")
        .def("get_idle_states", &plssvm::detail::tracking::cpu_idle_states_samples::get_idle_states, "the map of additional CPU idle states")
        .def("get_all_cpus_state_c0_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_all_cpus_state_c0_percent, "the percent of time all CPUs were in idle state c0")
        .def("get_any_cpu_state_c0_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_any_cpu_state_c0_percent, "the percent of time any CPU was in the idle state c0")
        .def("get_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_low_power_idle_state_percent, "the percent of time the CPUs was in the low power idle state")
        .def("get_system_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_system_low_power_idle_state_percent, "the percent of time the CPU was in the system low power idle state")
        .def("get_package_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_package_low_power_idle_state_percent, "the percent of time the CPU was in the package low power idle state")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_gfx_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuIdleStateSamples with\n{}\n>", self);
        });

    // bind the CPU hardware sampler class
    py::class_<plssvm::detail::tracking::cpu_hardware_sampler, plssvm::detail::tracking::hardware_sampler>(tracking_module, "CpuHardwareSampler")
        .def(py::init<>(), "construct a new CPU hardware sampler")
        .def(py::init<std::chrono::milliseconds>(), "construct a new CPU hardware sampler specifying the used sampling interval")
        .def("general_samples", &plssvm::detail::tracking::cpu_hardware_sampler::general_samples, "get all general samples")
        .def("clock_samples", &plssvm::detail::tracking::cpu_hardware_sampler::clock_samples, "get all clock related samples")
        .def("power_samples", &plssvm::detail::tracking::cpu_hardware_sampler::power_samples, "get all power related samples")
        .def("memory_samples", &plssvm::detail::tracking::cpu_hardware_sampler::memory_samples, "get all memory related samples")
        .def("temperature_samples", &plssvm::detail::tracking::cpu_hardware_sampler::temperature_samples, "get all temperature related samples")
        .def("gfx_samples", &plssvm::detail::tracking::cpu_hardware_sampler::gfx_samples, "get all gfx (iGPU) related samples")
        .def("idle_state_samples", &plssvm::detail::tracking::cpu_hardware_sampler::idle_state_samples, "get all idle state related samples")
        .def("__repr__", [](const plssvm::detail::tracking::cpu_hardware_sampler &self) {
            return fmt::format("<plssvm.detail.tracking.CpuHardwareSampler with\n{}\n>", self);
        });
}
