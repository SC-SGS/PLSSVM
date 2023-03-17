#include "plssvm/target_platforms.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_target_platforms(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::target_platform>(m, "target_platform")
        .value("automatic", plssvm::target_platform::automatic)
        .value("cpu", plssvm::target_platform::cpu)
        .value("gpu_nvidia", plssvm::target_platform::gpu_nvidia)
        .value("gpu_amd", plssvm::target_platform::gpu_amd)
        .value("gpu_intel", plssvm::target_platform::gpu_intel);

    // bind free functions
    m.def("list_available_target_platforms", &plssvm::list_available_target_platforms);
    m.def("determine_default_target_platform", &plssvm::determine_default_target_platform,
          py::arg("platform_device_list") = plssvm::list_available_target_platforms());
}