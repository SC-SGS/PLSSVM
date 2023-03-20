#include "plssvm/target_platforms.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_target_platforms(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::target_platform>(m, "TargetPlatform")
        .value("AUTOMATIC", plssvm::target_platform::automatic)
        .value("CPU", plssvm::target_platform::cpu)
        .value("GPU_NVIDIA", plssvm::target_platform::gpu_nvidia)
        .value("GPU_AMD", plssvm::target_platform::gpu_amd)
        .value("GPU_INTEL", plssvm::target_platform::gpu_intel);

    // bind free functions
    m.def("list_available_target_platforms", &plssvm::list_available_target_platforms);
    m.def("determine_default_target_platform", &plssvm::determine_default_target_platform,
          py::arg("platform_device_list") = plssvm::list_available_target_platforms());
}