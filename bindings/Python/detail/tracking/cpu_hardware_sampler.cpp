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
        .def("get_architecture", &plssvm::detail::tracking::cpu_general_samples::get_architecture)
        .def("get_byte_order", &plssvm::detail::tracking::cpu_general_samples::get_byte_order)
        .def("get_num_threads", &plssvm::detail::tracking::cpu_general_samples::get_num_threads)
        .def("get_threads_per_core", &plssvm::detail::tracking::cpu_general_samples::get_threads_per_core)
        .def("get_cores_per_socket", &plssvm::detail::tracking::cpu_general_samples::get_cores_per_socket)
        .def("get_num_sockets", &plssvm::detail::tracking::cpu_general_samples::get_num_sockets)
        .def("get_numa_nodes", &plssvm::detail::tracking::cpu_general_samples::get_numa_nodes)
        .def("get_vendor_id", &plssvm::detail::tracking::cpu_general_samples::get_vendor_id)
        .def("get_name", &plssvm::detail::tracking::cpu_general_samples::get_name)
        .def("get_flags", &plssvm::detail::tracking::cpu_general_samples::get_flags)
        .def("get_busy_percent", &plssvm::detail::tracking::cpu_general_samples::get_busy_percent)
        .def("get_ipc", &plssvm::detail::tracking::cpu_general_samples::get_ipc)
        .def("get_irq", &plssvm::detail::tracking::cpu_general_samples::get_irq)
        .def("get_smi", &plssvm::detail::tracking::cpu_general_samples::get_smi)
        .def("get_poll", &plssvm::detail::tracking::cpu_general_samples::get_poll)
        .def("get_poll_percent", &plssvm::detail::tracking::cpu_general_samples::get_poll_percent)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::cpu_clock_samples>(tracking_module, "CpuClockSamples")
        .def("get_frequency_boost", &plssvm::detail::tracking::cpu_clock_samples::get_frequency_boost)
        .def("get_min_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_min_frequency)
        .def("get_max_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_max_frequency)
        .def("get_average_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_average_frequency)
        .def("get_average_non_idle_frequency", &plssvm::detail::tracking::cpu_clock_samples::get_average_non_idle_frequency)
        .def("get_time_stamp_counter", &plssvm::detail::tracking::cpu_clock_samples::get_time_stamp_counter)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::cpu_power_samples>(tracking_module, "CpuPowerSamples")
        .def("get_package_watt", &plssvm::detail::tracking::cpu_power_samples::get_package_watt)
        .def("get_core_watt", &plssvm::detail::tracking::cpu_power_samples::get_core_watt)
        .def("get_ram_watt", &plssvm::detail::tracking::cpu_power_samples::get_ram_watt)
        .def("get_package_rapl_throttle_percent", &plssvm::detail::tracking::cpu_power_samples::get_package_rapl_throttle_percent)
        .def("get_dram_rapl_throttle_percent", &plssvm::detail::tracking::cpu_power_samples::get_dram_rapl_throttle_percent)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::cpu_memory_samples>(tracking_module, "CpuMemorySamples")
        .def("get_l1d_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l1d_cache)
        .def("get_l1i_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l1i_cache)
        .def("get_l2_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l2_cache)
        .def("get_l3_cache", &plssvm::detail::tracking::cpu_memory_samples::get_l3_cache)
        .def("get_memory_total", &plssvm::detail::tracking::cpu_memory_samples::get_memory_total)
        .def("get_swap_memory_total", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_total)
        .def("get_memory_used", &plssvm::detail::tracking::cpu_memory_samples::get_memory_used)
        .def("get_memory_free", &plssvm::detail::tracking::cpu_memory_samples::get_memory_free)
        .def("get_swap_memory_used", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_used)
        .def("get_swap_memory_free", &plssvm::detail::tracking::cpu_memory_samples::get_swap_memory_free)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::cpu_temperature_samples>(tracking_module, "CpuTemperatureSamples")
        .def("get_core_temperature", &plssvm::detail::tracking::cpu_temperature_samples::get_core_temperature)
        .def("get_core_throttle_percent", &plssvm::detail::tracking::cpu_temperature_samples::get_core_throttle_percent)
        .def("get_package_temperature", &plssvm::detail::tracking::cpu_temperature_samples::get_package_temperature)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_temperature_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuTemperatureSamples with\n{}\n>", self);
        });

    // bind the gfx samples
    py::class_<plssvm::detail::tracking::cpu_gfx_samples>(tracking_module, "CpuGfxSamples")
        .def("get_gfx_render_state_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_render_state_percent)
        .def("get_gfx_frequency", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_frequency)
        .def("get_average_gfx_frequency", &plssvm::detail::tracking::cpu_gfx_samples::get_average_gfx_frequency)
        .def("get_gfx_state_c0_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_state_c0_percent)
        .def("get_cpu_works_for_gpu_percent", &plssvm::detail::tracking::cpu_gfx_samples::get_cpu_works_for_gpu_percent)
        .def("get_gfx_watt", &plssvm::detail::tracking::cpu_gfx_samples::get_gfx_watt)
        .def("__repr__", [](const plssvm::detail::tracking::cpu_gfx_samples &self) {
            return fmt::format("<plssvm.detail.tracking.CpuGfxSamples with\n{}\n>", self);
        });

    // bind the idle state samples
    py::class_<plssvm::detail::tracking::cpu_idle_states_samples>(tracking_module, "CpuIdleStateSamples")
        .def("get_idle_states", &plssvm::detail::tracking::cpu_idle_states_samples::get_idle_states)
        .def("get_all_cpus_state_c0_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_all_cpus_state_c0_percent)
        .def("get_any_cpu_state_c0_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_any_cpu_state_c0_percent)
        .def("get_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_low_power_idle_state_percent)
        .def("get_system_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_system_low_power_idle_state_percent)
        .def("get_package_low_power_idle_state_percent", &plssvm::detail::tracking::cpu_idle_states_samples::get_package_low_power_idle_state_percent)
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
