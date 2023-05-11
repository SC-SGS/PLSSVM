/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/target_platforms.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types: std::vector

namespace py = pybind11;

void init_target_platforms(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::target_platform>(m, "TargetPlatform")
        .value("AUTOMATIC", plssvm::target_platform::automatic, "the default target with respect to the used backend type; checks for available devices in the following order: NVIDIA GPUs -> AMD GPUs -> Intel GPUs -> CPUs")
        .value("CPU", plssvm::target_platform::cpu, "target CPUs only (Intel, AMD, IBM, ...)")
        .value("GPU_NVIDIA", plssvm::target_platform::gpu_nvidia, "target GPUs from NVIDIA")
        .value("GPU_AMD", plssvm::target_platform::gpu_amd, "target GPUs from AMD")
        .value("GPU_INTEL", plssvm::target_platform::gpu_intel, "target GPUs from Intel");

    // bind free functions
    m.def("list_available_target_platforms", &plssvm::list_available_target_platforms, "list the available target platforms (as defined during CMake configuration)");
    m.def("determine_default_target_platform", &plssvm::determine_default_target_platform, "determine the default target platform given the list of available target platforms", py::arg("platform_device_list") = plssvm::list_available_target_platforms());
}