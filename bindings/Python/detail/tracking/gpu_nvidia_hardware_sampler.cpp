/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_nvidia/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_nvidia_hardware_sampler
#include "plssvm/detail/tracking/gpu_nvidia/nvml_samples.hpp"      // plssvm::detail::tracking::{nvml_general_samples, nvml_clock_samples, nvml_power_samples, nvml_memory_samples, nvml_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"             // plssvm::detail::tracking::hardware_sampler

#include "fmt/core.h"           // fmt::format
#include "pybind11/chrono.h"    // automatic bindings for std::chrono::milliseconds
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // automatic bindings for std::optional and std::vector

#include <chrono>   // std::chrono::milliseconds
#include <cstddef>  // std::size_t

namespace py = pybind11;

void init_gpu_nvidia_hardware_sampler(py::module_ &m) {
    // use a detail.tracking submodule for the hardware sampling
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    const py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");

    // bind the general samples
    py::class_<plssvm::detail::tracking::nvml_general_samples>(tracking_module, "NvmlGeneralSamples")
        .def("get_name", &plssvm::detail::tracking::nvml_general_samples::get_name)
        .def("get_persistence_mode", &plssvm::detail::tracking::nvml_general_samples::get_persistence_mode)
        .def("get_num_cores", &plssvm::detail::tracking::nvml_general_samples::get_num_cores)
        .def("get_performance_state", &plssvm::detail::tracking::nvml_general_samples::get_performance_state)
        .def("get_utilization_gpu", &plssvm::detail::tracking::nvml_general_samples::get_utilization_gpu)
        .def("get_utilization_mem", &plssvm::detail::tracking::nvml_general_samples::get_utilization_mem)
        .def("__repr__", [](const plssvm::detail::tracking::nvml_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.NvmlGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::nvml_clock_samples>(tracking_module, "NvmlClockSamples")
        .def("get_adaptive_clock_status", &plssvm::detail::tracking::nvml_clock_samples::get_adaptive_clock_status)
        .def("get_clock_graph_min", &plssvm::detail::tracking::nvml_clock_samples::get_clock_graph_min)
        .def("get_clock_graph_max", &plssvm::detail::tracking::nvml_clock_samples::get_clock_graph_max)
        .def("get_clock_sm_max", &plssvm::detail::tracking::nvml_clock_samples::get_clock_sm_max)
        .def("get_clock_mem_min", &plssvm::detail::tracking::nvml_clock_samples::get_clock_mem_min)
        .def("get_clock_mem_max", &plssvm::detail::tracking::nvml_clock_samples::get_clock_mem_max)
        .def("get_clock_graph", &plssvm::detail::tracking::nvml_clock_samples::get_clock_graph)
        .def("get_clock_sm", &plssvm::detail::tracking::nvml_clock_samples::get_clock_sm)
        .def("get_clock_mem", &plssvm::detail::tracking::nvml_clock_samples::get_clock_mem)
        .def("get_clock_throttle_reason", &plssvm::detail::tracking::nvml_clock_samples::get_clock_throttle_reason)
        .def("get_auto_boosted_clocks", &plssvm::detail::tracking::nvml_clock_samples::get_auto_boosted_clocks)
        .def("__repr__", [](const plssvm::detail::tracking::nvml_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.NvmlClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::nvml_power_samples>(tracking_module, "NvmlPowerSamples")
        .def("get_power_management_limit", &plssvm::detail::tracking::nvml_power_samples::get_power_management_limit)
        .def("get_power_enforced_limit", &plssvm::detail::tracking::nvml_power_samples::get_power_enforced_limit)
        .def("get_power_state", &plssvm::detail::tracking::nvml_power_samples::get_power_state)
        .def("get_power_usage", &plssvm::detail::tracking::nvml_power_samples::get_power_usage)
        .def("get_power_total_energy_consumption", &plssvm::detail::tracking::nvml_power_samples::get_power_total_energy_consumption)
        .def("__repr__", [](const plssvm::detail::tracking::nvml_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.NvmlPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::nvml_memory_samples>(tracking_module, "NvmlMemorySamples")
        .def("get_memory_total", &plssvm::detail::tracking::nvml_memory_samples::get_memory_total)
        .def("get_pcie_link_max_speed", &plssvm::detail::tracking::nvml_memory_samples::get_pcie_link_max_speed)
        .def("get_memory_bus_width", &plssvm::detail::tracking::nvml_memory_samples::get_memory_bus_width)
        .def("get_max_pcie_link_generation", &plssvm::detail::tracking::nvml_memory_samples::get_max_pcie_link_generation)
        .def("get_memory_free", &plssvm::detail::tracking::nvml_memory_samples::get_memory_free)
        .def("get_memory_used", &plssvm::detail::tracking::nvml_memory_samples::get_memory_used)
        .def("get_pcie_link_speed", &plssvm::detail::tracking::nvml_memory_samples::get_pcie_link_speed)
        .def("get_pcie_link_width", &plssvm::detail::tracking::nvml_memory_samples::get_pcie_link_width)
        .def("get_pcie_link_generation", &plssvm::detail::tracking::nvml_memory_samples::get_pcie_link_generation)
        .def("__repr__", [](const plssvm::detail::tracking::nvml_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.NvmlMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::nvml_temperature_samples>(tracking_module, "NvmlTemperatureSamples")
        .def("get_num_fans", &plssvm::detail::tracking::nvml_temperature_samples::get_num_fans)
        .def("get_min_fan_speed", &plssvm::detail::tracking::nvml_temperature_samples::get_min_fan_speed)
        .def("get_max_fan_speed", &plssvm::detail::tracking::nvml_temperature_samples::get_max_fan_speed)
        .def("get_temperature_threshold_gpu_max", &plssvm::detail::tracking::nvml_temperature_samples::get_temperature_threshold_gpu_max)
        .def("get_temperature_threshold_mem_max", &plssvm::detail::tracking::nvml_temperature_samples::get_temperature_threshold_mem_max)
        .def("get_fan_speed", &plssvm::detail::tracking::nvml_temperature_samples::get_fan_speed)
        .def("get_temperature_gpu", &plssvm::detail::tracking::nvml_temperature_samples::get_temperature_gpu)
        .def("__repr__", [](const plssvm::detail::tracking::nvml_temperature_samples &self) {
            return fmt::format("<plssvm.detail.tracking.NvmlTemperatureSamples with\n{}\n>", self);
        });

    // bind the GPU NVIDIA hardware sampler class
    py::class_<plssvm::detail::tracking::gpu_nvidia_hardware_sampler, plssvm::detail::tracking::hardware_sampler>(tracking_module, "GpuNvidiaHardwareSampler")
        .def(py::init<std::size_t>(), "construct a new NVIDIA GPU hardware sampler specifying the device to sample")
        .def(py::init<std::size_t, std::chrono::milliseconds>(), "construct a new NVIDIA GPU hardware sampler specifying the device to sample and the used sampling interval")
        .def("general_samples", &plssvm::detail::tracking::gpu_nvidia_hardware_sampler::general_samples, "get all general samples")
        .def("clock_samples", &plssvm::detail::tracking::gpu_nvidia_hardware_sampler::clock_samples, "get all clock related samples")
        .def("power_samples", &plssvm::detail::tracking::gpu_nvidia_hardware_sampler::power_samples, "get all power related samples")
        .def("memory_samples", &plssvm::detail::tracking::gpu_nvidia_hardware_sampler::memory_samples, "get all memory related samples")
        .def("temperature_samples", &plssvm::detail::tracking::gpu_nvidia_hardware_sampler::temperature_samples, "get all temperature related samples")
        .def("__repr__", [](const plssvm::detail::tracking::gpu_nvidia_hardware_sampler &self) {
            return fmt::format("<plssvm.detail.tracking.GpuNvidiaHardwareSampler with\n{}\n>", self);
        });
}
