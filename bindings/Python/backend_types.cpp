#include "plssvm/backend_types.hpp"

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_backend_types(py::module_ & m) {
    // bind enum class
    py::enum_<plssvm::backend_type>(m, "BackendType")
        .value("AUTOMATIC", plssvm::backend_type::automatic)
        .value("OPENMP", plssvm::backend_type::openmp)
        .value("CUDA", plssvm::backend_type::cuda)
        .value("HIP", plssvm::backend_type::hip)
        .value("OPENCL", plssvm::backend_type::opencl)
        .value("SYCL", plssvm::backend_type::sycl);

    // bind free functions
    m.def("list_available_backends", &plssvm::list_available_backends);
    m.def("determine_default_backend", &plssvm::determine_default_backend,
          py::arg("available_backends") = plssvm::list_available_backends(),
          py::arg("available_target_platforms") = plssvm::list_available_target_platforms());
}