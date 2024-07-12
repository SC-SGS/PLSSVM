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
        .def("get_name", &plssvm::detail::tracking::rocm_smi_general_samples::get_name, "the name of the device")
        .def("get_performance_level", &plssvm::detail::tracking::rocm_smi_general_samples::get_performance_level, "the performance level: one of rsmi_dev_perf_level_t")
        .def("get_utilization_gpu", &plssvm::detail::tracking::rocm_smi_general_samples::get_utilization_gpu, "the GPU compute utilization in percent")
        .def("get_utilization_mem", &plssvm::detail::tracking::rocm_smi_general_samples::get_utilization_mem, "the GPU memory utilization in percent")
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::rocm_smi_clock_samples>(tracking_module, "RocmSmiClockSamples")
        .def("get_clock_system_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system_min, "the minimum possible system clock frequency in Hz")
        .def("get_clock_system_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system_max, "the maximum possible system clock frequency in Hz")
        .def("get_clock_socket_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket_min, "the minimum possible socket clock frequency in Hz")
        .def("get_clock_socket_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket_max, "the maximum possible socket clock frequency in Hz")
        .def("get_clock_memory_min", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory_min, "the minimum possible memory clock frequency in Hz")
        .def("get_clock_memory_max", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory_max, "the maximum possible memory clock frequency in Hz")
        .def("get_clock_system", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_system, "the current system clock frequency in Hz")
        .def("get_clock_socket", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_socket, "the current socket clock frequency in Hz")
        .def("get_clock_memory", &plssvm::detail::tracking::rocm_smi_clock_samples::get_clock_memory, "the current memory clock frequency in Hz")
        .def("get_overdrive_level", &plssvm::detail::tracking::rocm_smi_clock_samples::get_overdrive_level, "the GPU overdrive percentage")
        .def("get_memory_overdrive_level", &plssvm::detail::tracking::rocm_smi_clock_samples::get_memory_overdrive_level, "the GPU's memory overdrive percentage")
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::rocm_smi_power_samples>(tracking_module, "RocmSmiPowerSamples")
        .def("get_power_default_cap", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_default_cap, "the default power cap, may be different from power cap")
        .def("get_power_cap", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_cap, "if the GPU draws more power (μW) than the power cap, the GPU may throttle")
        .def("get_power_type", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_type, "the type of the power management: either current power draw or average power draw")
        .def("get_available_power_profiles", &plssvm::detail::tracking::rocm_smi_power_samples::get_available_power_profiles, "a list of the available power profiles")
        .def("get_power_usage", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_usage, "the current GPU socket power draw in μW")
        .def("get_power_total_energy_consumption", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_total_energy_consumption, "the total power consumption since the last driver reload in μJ")
        .def("get_power_profile", &plssvm::detail::tracking::rocm_smi_power_samples::get_power_profile, "the current active power profile; one of 'available_power_profiles'")
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::rocm_smi_memory_samples>(tracking_module, "RocmSmiMemorySamples")
        .def("get_memory_total", &plssvm::detail::tracking::rocm_smi_memory_samples::get_memory_total, "the total available memory in Byte")
        .def("get_visible_memory_total", &plssvm::detail::tracking::rocm_smi_memory_samples::get_visible_memory_total, "the total visible available memory in Byte, may be smaller than the total memory")
        .def("get_min_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_min_num_pcie_lanes, "the minimum number of used PCIe lanes")
        .def("get_max_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_max_num_pcie_lanes, "the maximum number of used PCIe lanes")
        .def("get_memory_used", &plssvm::detail::tracking::rocm_smi_memory_samples::get_memory_used, "the currently used memory in Byte")
        .def("get_pcie_transfer_rate", &plssvm::detail::tracking::rocm_smi_memory_samples::get_pcie_transfer_rate, "the current PCIe transfer rate in T/s")
        .def("get_num_pcie_lanes", &plssvm::detail::tracking::rocm_smi_memory_samples::get_num_pcie_lanes, "the number of currently used PCIe lanes")
        .def("__repr__", [](const plssvm::detail::tracking::rocm_smi_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.RocmSmiMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::rocm_smi_temperature_samples>(tracking_module, "RocmSmiTemperatureSamples")
        .def("get_num_fans", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_num_fans, "the number of fans (if any)")
        .def("get_max_fan_speed", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_max_fan_speed, "the maximum fan speed")
        .def("get_temperature_edge_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge_min, "the minimum temperature on the GPU's edge temperature sensor in m°C")
        .def("get_temperature_edge_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge_max, "the maximum temperature on the GPU's edge temperature sensor in m°C")
        .def("get_temperature_hotspot_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot_min, "the minimum temperature on the GPU's hotspot temperature sensor in m°C")
        .def("get_temperature_hotspot_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot_max, "the maximum temperature on the GPU's hotspot temperature sensor in m°C")
        .def("get_temperature_memory_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory_min, "the minimum temperature on the GPU's memory temperature sensor in m°C")
        .def("get_temperature_memory_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory_max, "the maximum temperature on the GPU's memory temperature sensor in m°C")
        .def("get_temperature_hbm_0_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0_min, "the minimum temperature on the GPU's HBM0 temperature sensor in m°C")
        .def("get_temperature_hbm_0_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0_max, "the maximum temperature on the GPU's HBM0 temperature sensor in m°C")
        .def("get_temperature_hbm_1_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1_min, "the minimum temperature on the GPU's HBM1 temperature sensor in m°C")
        .def("get_temperature_hbm_1_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1_max, "the maximum temperature on the GPU's HBM1 temperature sensor in m°C")
        .def("get_temperature_hbm_2_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2_min, "the minimum temperature on the GPU's HBM2 temperature sensor in m°C")
        .def("get_temperature_hbm_2_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2_max, "the maximum temperature on the GPU's HBM2 temperature sensor in m°C")
        .def("get_temperature_hbm_3_min", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3_min, "the minimum temperature on the GPU's HBM3 temperature sensor in m°C")
        .def("get_temperature_hbm_3_max", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3_max, "the maximum temperature on the GPU's HBM3 temperature sensor in m°C")
        .def("get_fan_speed", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_fan_speed, "the current fan speed in %")
        .def("get_temperature_edge", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_edge, "the current temperature on the GPU's edge temperature sensor in m°C")
        .def("get_temperature_hotspot", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hotspot, "the current temperature on the GPU's hotspot temperature sensor in m°C")
        .def("get_temperature_memory", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_memory, "the current temperature on the GPU's memory temperature sensor in m°C")
        .def("get_temperature_hbm_0", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_0, "the current temperature on the GPU's HBM0 temperature sensor in m°C")
        .def("get_temperature_hbm_1", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_1, "the current temperature on the GPU's HBM1 temperature sensor in m°C")
        .def("get_temperature_hbm_2", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_2, "the current temperature on the GPU's HBM2 temperature sensor in m°C")
        .def("get_temperature_hbm_3", &plssvm::detail::tracking::rocm_smi_temperature_samples::get_temperature_hbm_3, "the current temperature on the GPU's HBM3 temperature sensor in m°C")
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
