/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types: std::vector

namespace py = pybind11;

void init_backend_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::backend_type>(m, "BackendType")
        .value("AUTOMATIC", plssvm::backend_type::automatic, "the default backend; depends on the specified target platform")
        .value("OPENMP", plssvm::backend_type::openmp, "OpenMP to target CPUs only (currently no OpenMP target offloading support)")
        .value("CUDA", plssvm::backend_type::cuda, "CUDA to target NVIDIA GPUs only")
        .value("HIP", plssvm::backend_type::hip, "HIP to target AMD and NVIDIA GPUs")
        .value("OPENCL", plssvm::backend_type::opencl, "OpenCL to target CPUs and GPUs from different vendors")
        .value("SYCL", plssvm::backend_type::sycl, "SYCL o target CPUs and GPUs from different vendors; currently tested SYCL implementations are DPC++ and AdaptiveCpp");

    // bind free functions
    m.def("list_available_backends", &plssvm::list_available_backends, "list the available backends (as found during CMake configuration)");
    m.def("determine_default_backend", &plssvm::determine_default_backend, "determine the default backend given the list of available backends and target platforms", py::arg("available_backends") = plssvm::list_available_backends(), py::arg("available_target_platforms") = plssvm::list_available_target_platforms());
}