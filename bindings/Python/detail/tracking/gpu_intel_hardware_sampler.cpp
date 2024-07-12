/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_intel/hardware_sampler.hpp"    // plssvm::detail::tracking::gpu_intel_hardware_sampler
#include "plssvm/detail/tracking/gpu_intel/level_zero_samples.hpp"  // plssvm::detail::tracking::{level_zero_general_samples, level_zero_clock_samples, level_zero_power_samples, level_zero_memory_samples, level_zero_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"              // plssvm::detail::tracking::hardware_sampler

#include "fmt/core.h"           // fmt::format
#include "pybind11/chrono.h"    // automatic bindings for std::chrono::milliseconds
#include "pybind11/pybind11.h"  // py::module_
#include "pybind11/stl.h"       // automatic bindings for std::optional and std::vector

#include <chrono>   // std::chrono::milliseconds
#include <cstddef>  // std::size_t

namespace py = pybind11;

void init_gpu_intel_hardware_sampler(py::module_ &m) {
    // use a detail.tracking submodule for the hardware sampling
    py::module_ detail_module = m.def_submodule("detail", "a module containing detail functionality");
    const py::module_ tracking_module = detail_module.def_submodule("tracking", "a module containing performance tracking and hardware sampling functionality");

    // bind the general samples
    py::class_<plssvm::detail::tracking::level_zero_general_samples>(tracking_module, "LevelZeroGeneralSamples")
        .def("get_name", &plssvm::detail::tracking::level_zero_general_samples::get_name, "the model name of the device")
        .def("get_standby_mode", &plssvm::detail::tracking::level_zero_general_samples::get_standby_mode, "the enabled standby mode (power saving or never)")
        .def("get_num_threads_per_eu", &plssvm::detail::tracking::level_zero_general_samples::get_num_threads_per_eu, "the number of threads per EU unit")
        .def("get_eu_simd_width", &plssvm::detail::tracking::level_zero_general_samples::get_eu_simd_width, "the physical EU unit SIMD width")
        .def("__repr__", [](const plssvm::detail::tracking::level_zero_general_samples &self) {
            return fmt::format("<plssvm.detail.tracking.LevelZeroGeneralSamples with\n{}\n>", self);
        });

    // bind the clock samples
    py::class_<plssvm::detail::tracking::level_zero_clock_samples>(tracking_module, "LevelZeroClockSamples")
        .def("get_clock_gpu_min", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_gpu_min, "the minimum possible GPU clock frequency in MHz")
        .def("get_clock_gpu_max", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_gpu_max, "the maximum possible GPU clock frequency in MHz")
        .def("get_available_clocks_gpu", &plssvm::detail::tracking::level_zero_clock_samples::get_available_clocks_gpu, "the available GPU clock frequencies in MHz (slowest to fastest)")
        .def("get_clock_mem_min", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_mem_min, "the minimum possible memory clock frequency in MHz")
        .def("get_clock_mem_max", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_mem_max, "the maximum possible memory clock frequency in MHz")
        .def("get_available_clocks_mem", &plssvm::detail::tracking::level_zero_clock_samples::get_available_clocks_mem, "the available memory clock frequencies in MHz (slowest to fastest)")
        .def("get_tdp_frequency_limit_gpu", &plssvm::detail::tracking::level_zero_clock_samples::get_tdp_frequency_limit_gpu, "the current maximum allowed GPU frequency based on the TDP limit in MHz")
        .def("get_clock_gpu", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_gpu, "the current GPU frequency in MHz")
        .def("get_throttle_reason_gpu", &plssvm::detail::tracking::level_zero_clock_samples::get_throttle_reason_gpu, "the current GPU frequency throttle reason")
        .def("get_tdp_frequency_limit_mem", &plssvm::detail::tracking::level_zero_clock_samples::get_tdp_frequency_limit_mem, "the current maximum allowed memory frequency based on the TDP limit in MHz")
        .def("get_clock_mem", &plssvm::detail::tracking::level_zero_clock_samples::get_clock_mem, "the current memory frequency in MHz")
        .def("get_throttle_reason_mem", &plssvm::detail::tracking::level_zero_clock_samples::get_throttle_reason_mem, "the current memory frequency throttle reason")
        .def("__repr__", [](const plssvm::detail::tracking::level_zero_clock_samples &self) {
            return fmt::format("<plssvm.detail.tracking.LevelZeroClockSamples with\n{}\n>", self);
        });

    // bind the power samples
    py::class_<plssvm::detail::tracking::level_zero_power_samples>(tracking_module, "LevelZeroPowerSamples")
        .def("get_energy_threshold_enabled", &plssvm::detail::tracking::level_zero_power_samples::get_energy_threshold_enabled, "true if the energy threshold is enabled")
        .def("get_energy_threshold", &plssvm::detail::tracking::level_zero_power_samples::get_energy_threshold, "the energy threshold in J")
        .def("get_power_total_energy_consumption", &plssvm::detail::tracking::level_zero_power_samples::get_power_total_energy_consumption, "the total power consumption since the last driver reload in mJ")
        .def("__repr__", [](const plssvm::detail::tracking::level_zero_power_samples &self) {
            return fmt::format("<plssvm.detail.tracking.LevelZeroPowerSamples with\n{}\n>", self);
        });

    // bind the memory samples
    py::class_<plssvm::detail::tracking::level_zero_memory_samples>(tracking_module, "LevelZeroMemorySamples")
        .def("get_memory_total", &plssvm::detail::tracking::level_zero_memory_samples::get_memory_total, "the total memory size of the different memory modules in Bytes")
        .def("get_allocatable_memory_total", &plssvm::detail::tracking::level_zero_memory_samples::get_allocatable_memory_total, "the total allocatable memory size of the different memory modules in Bytes")
        .def("get_pcie_link_max_speed", &plssvm::detail::tracking::level_zero_memory_samples::get_pcie_link_max_speed, "the maximum PCIe bandwidth in bytes/sec")
        .def("get_pcie_max_width", &plssvm::detail::tracking::level_zero_memory_samples::get_pcie_max_width, "the PCIe lane width")
        .def("get_max_pcie_link_generation", &plssvm::detail::tracking::level_zero_memory_samples::get_max_pcie_link_generation, "the PCIe generation")
        .def("get_bus_width", &plssvm::detail::tracking::level_zero_memory_samples::get_bus_width, "the bus width of the different memory modules")
        .def("get_num_channels", &plssvm::detail::tracking::level_zero_memory_samples::get_num_channels, "the number of memory channels of the different memory modules")
        .def("get_location", &plssvm::detail::tracking::level_zero_memory_samples::get_location, "the location of the different memory modules (system or device)")
        .def("get_memory_free", &plssvm::detail::tracking::level_zero_memory_samples::get_memory_free, "the currently free memory of the different memory modules in Bytes")
        .def("get_pcie_link_speed", &plssvm::detail::tracking::level_zero_memory_samples::get_pcie_link_speed, "the current PCIe bandwidth in bytes/sec")
        .def("get_pcie_link_width", &plssvm::detail::tracking::level_zero_memory_samples::get_pcie_link_width, "the current PCIe lane width")
        .def("get_pcie_link_generation", &plssvm::detail::tracking::level_zero_memory_samples::get_pcie_link_generation, "the current PCIe generation")
        .def("__repr__", [](const plssvm::detail::tracking::level_zero_memory_samples &self) {
            return fmt::format("<plssvm.detail.tracking.LevelZeroMemorySamples with\n{}\n>", self);
        });

    // bind the temperature samples
    py::class_<plssvm::detail::tracking::level_zero_temperature_samples>(tracking_module, "LevelZeroTemperatureSamples")
        .def("get_temperature_max", &plssvm::detail::tracking::level_zero_temperature_samples::get_temperature_max, "the maximum temperature for the sensor in °C")
        .def("get_temperature_psu", &plssvm::detail::tracking::level_zero_temperature_samples::get_temperature_psu, "the temperature of the PSU in °C")
        .def("get_temperature", &plssvm::detail::tracking::level_zero_temperature_samples::get_temperature, "the current temperature for the sensor in °C")
        .def("__repr__", [](const plssvm::detail::tracking::level_zero_temperature_samples &self) {
            return fmt::format("<plssvm.detail.tracking.LevelZeroTemperatureSamples with\n{}\n>", self);
        });

    // bind the GPU Intel hardware sampler class
    py::class_<plssvm::detail::tracking::gpu_intel_hardware_sampler, plssvm::detail::tracking::hardware_sampler>(tracking_module, "GpuIntelHardwareSampler")
        .def(py::init<std::size_t>(), "construct a new Intel GPU hardware sampler specifying the device to sample")
        .def(py::init<std::size_t, std::chrono::milliseconds>(), "construct a new Intel GPU hardware sampler specifying the device to sample and the used sampling interval")
        .def("general_samples", &plssvm::detail::tracking::gpu_intel_hardware_sampler::general_samples, "get all general samples")
        .def("clock_samples", &plssvm::detail::tracking::gpu_intel_hardware_sampler::clock_samples, "get all clock related samples")
        .def("power_samples", &plssvm::detail::tracking::gpu_intel_hardware_sampler::power_samples, "get all power related samples")
        .def("memory_samples", &plssvm::detail::tracking::gpu_intel_hardware_sampler::memory_samples, "get all memory related samples")
        .def("temperature_samples", &plssvm::detail::tracking::gpu_intel_hardware_sampler::temperature_samples, "get all temperature related samples")
        .def("__repr__", [](const plssvm::detail::tracking::gpu_intel_hardware_sampler &self) {
            return fmt::format("<plssvm.detail.tracking.GpuIntelHardwareSampler with\n{}\n>", self);
        });
}
