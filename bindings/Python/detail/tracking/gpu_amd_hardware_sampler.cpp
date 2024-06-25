/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_amd/hardware_sampler.hpp"  // plssvm::detail::tracking::gpu_amd_hardware_sampler
#include "plssvm/detail/tracking/gpu_amd/rocm_smi_samples.hpp"  // plssvm::detail::tracking::{rocm_smi_general_samples, rocm_smi_clock_samples, rocm_smi_power_samples, rocm_smi_memory_samples, rocm_smi_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler

#include "fmt/core.h"           // fmt::format
#include "pybind11/chrono.h"    // automatic bindings for std::chrono::milliseconds
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // automatic bindings for std::optional and std::vector

#include <chrono>   // std::chrono::milliseconds
#include <cstddef>  // std::size_t

namespace py = pybind11;

void init_gpu_amd_hardware_sampler(py::module_ &m) {
    // use a detail.tracking submodule for the hardware sampling
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    const py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");

    // bind the general samples
    py::class_<plssvm::detail::tracking::rocm_smi_general_samples>(tracking_module, "RocmSmiGeneralSamples")
        .def("get_name", &plssvm::detail::tracking::rocm_smi_general_samples::get_name)
        .def("get_performance_level", &plssvm::detail::tracking::rocm_smi_general_samples::get_performance_level)
        .def("get_utilization_gpu", &plssvm::detail::tracking::rocm_smi_general_samples::get_utilization_gpu)
        .def("get_utilization_mem", &plssvm::detail::tracking::rocm_smi_general_samples::get_utilization_mem)
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::rocm_smi_clock_samples>(tracking_module, "RocmSmiClockSamples")
        .def("get_clock_system_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system_min)
        .def("get_clock_system_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system_max)
        .def("get_clock_socket_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket_min)
        .def("get_clock_socket_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket_max)
        .def("get_clock_memory_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory_min)
        .def("get_clock_memory_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory_max)
        .def("get_clock_system", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system)
        .def("get_clock_socket", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket)
        .def("get_clock_memory", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory)
        .def("get_clock_throttle_status", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_throttle_status)
        .def("get_overdrive_level", &plssvm::detail::tracking::rocm_smi_clock_samples::get_overdrive_level)
        .def("get_memory_overdrive_level", &plssvm::detail::tracking::rocm_smi_clock_samples::get_memory_overdrive_level)
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::rocm_smi_power_samples>(tracking_module, "RocmSmiPowerSamples")
        .def("get_power_default_cap", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_default_cap)
        .def("get_power_cap", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_cap)
        .def("get_power_type", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_type)
        .def("get_available_power_profiles", &plssvm::detail::tracking::rocm_smi_power_samples::get_available_power_profiles)
        .def("get_power_usage", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_usage)
        .def("get_power_total_energy_consumption", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_total_energy_consumption)
        .def("get_power_profile", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_profile)
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::rocm_smi_memory_samples>(tracking_module, "RocmSmiMemorySamples")
        .def("get_memory_total", &plssvm::detail::tracking::rocm_smi_memory_samples::get_memory_total)
        .def("get_visible_memory_total", &plssvm::detail::tracking::rocm_smi_memory_samples::get_visible_memory_total)
        .def("get_min_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_min_num_pcie_lanes)
        .def("get_max_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_max_num_pcie_lanes)
        .def("get_memory_used", &plssvm::detail::tracking::rocm_smi_memory_samples::get_memory_used)
        .def("get_pcie_transfer_rate", &plssvm::detail::tracking::rocm_smi_memory_samples::get_pcie_transfer_rate)
        .def("get_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_num_pcie_lanes)
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::rocm_smi_temperature_samples>(tracking_module, "RocmSmiTemperatureSamples")
        .def("get_num_fans", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_num_fans)
        .def("get_max_fan_speed", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_max_fan_speed)
        .def("get_temperature_edge_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge_min)
        .def("get_temperature_edge_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge_max)
        .def("get_temperature_hotspot_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot_min)
        .def("get_temperature_hotspot_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot_max)
        .def("get_temperature_memory_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory_min)
        .def("get_temperature_memory_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory_max)
        .def("get_temperature_hbm_0_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0_min)
        .def("get_temperature_hbm_0_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0_max)
        .def("get_temperature_hbm_1_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1_min)
        .def("get_temperature_hbm_1_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1_max)
        .def("get_temperature_hbm_2_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2_min)
        .def("get_temperature_hbm_2_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2_max)
        .def("get_temperature_hbm_3_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3_min)
        .def("get_temperature_hbm_3_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3_max)
        .def("get_fan_speed", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_fan_speed)
        .def("get_temperature_edge", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge)
        .def("get_temperature_hotspot", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot)
        .def("get_temperature_memory", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory)
        .def("get_temperature_hbm_0", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0)
        .def("get_temperature_hbm_1", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1)
        .def("get_temperature_hbm_2", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2)
        .def("get_temperature_hbm_3", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3)
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_temperature_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiTemperatureSamples with\n{}\n>", self);
        });

    // bind the GPU AMD hardware sampler class
    py::class_<plssvm::detail::tracking::gpu_amd_hardware_sampler, plssvm::detail::tracking::hardware_sampler>(tracking_module, "GpuAmdHardwareSampler")
        .def(py::init<std::size_t>(), "construct a new AMD GPU hardware sampler specifying the device to sample")
        .def(py::init<std::size_t, std::chrono::milliseconds>(), "construct a new AMD GPU hardware sampler specifying the device to sample and the used sampling interval")
        .def("general_samples", &plssvm::detail::tracking::gpu_amd_hardware_sampler::general_samples, "get all general samples")
        .def("clock_samples", &plssvm::detail::tracking::gpu_amd_hardware_sampler::clock_samples, "get all clock related samples")
        .def("power_samples", &plssvm::detail::tracking::gpu_amd_hardware_sampler::power_samples, "get all power related samples")
        .def("memory_samples", &plssvm::detail::tracking::gpu_amd_hardware_sampler::memory_samples, "get all memory related samples")
        .def("temperature_samples", &plssvm::detail::tracking::gpu_amd_hardware_sampler::temperature_samples, "get all temperature related samples")
        .def("__repr__", [](const plssvm::detail::tracking::gpu_amd_hardware_sampler &self) {
            return fmt::format("<plssvm.detail.tracking.GpuAmdHardwareSampler with\n{}\n>", self);
        });
}
