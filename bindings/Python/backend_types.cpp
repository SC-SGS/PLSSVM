/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"  // plssvm::backend_type, plssvm::list_available_backends, plssvm::determine_default_backend

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings
#include "pybind11/native_enum.h"  // py::native_enum
#include "pybind11/pybind11.h"     // py::module_
#include "pybind11/stl.h"          // support for STL types: std::vector

#include <vector>  // std::vector

namespace py = pybind11;

void init_backend_types(py::module_ &m) {
    // bind enum class
    py::native_enum<plssvm::backend_type> py_enum(m, "BackendType", "enum.Enum", "Enum class for all possible backend types, all different SYCL implementations have the same backend type \"sycl\".");
    py_enum
        .value("AUTOMATIC", plssvm::backend_type::automatic, "the default backend; depends on the specified target platform")
        .value("OPENMP", plssvm::backend_type::openmp, "OpenMP to target CPUs only (currently no OpenMP target offloading support)")
        .value("HPX", plssvm::backend_type::hpx, "HPX to target CPUs only (currently no GPU executor support)")
        .value("STDPAR", plssvm::backend_type::stdpar, "C++ standard parallelism to target CPUs and GPUs from different vendors based on the used stdpar implementation; supported implementations are: nvhpc (nvc++), roc-stdpar, AdaptiveCpp, Intel LLVM (icpx), and GNU GCC + TBB")
        .value("CUDA", plssvm::backend_type::cuda, "CUDA to target NVIDIA GPUs only")
        .value("HIP", plssvm::backend_type::hip, "HIP to target AMD and NVIDIA GPUs")
        .value("OPENCL", plssvm::backend_type::opencl, "OpenCL to target CPUs and GPUs from different vendors")
        .value("SYCL", plssvm::backend_type::sycl, "SYCL to target CPUs and GPUs from different vendors; currently tested SYCL implementations are DPC++ and AdaptiveCpp")
        .value("KOKKOS", plssvm::backend_type::kokkos, "Kokkos to target CPUs and GPUs from different vendors; currently all Kokkos execution spaces except Kokkos::Experimental::OpenMPTarget and Kokkos::Experimental::OpenACC are supported")
        .finalize();

    // bind free functions
    m.def("list_available_backends", &plssvm::list_available_backends, "list the available backends (as found during CMake configuration)");
    m.def("determine_default_backend", &plssvm::determine_default_backend, "determine the default backend given the list of available backends and target platforms", py::arg("available_backends") = plssvm::list_available_backends(), py::arg("available_target_platforms") = plssvm::list_available_target_platforms());
}
