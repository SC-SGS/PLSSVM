#include "plssvm/backend_types.hpp"

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_backend_types(py::module & m) {
    // bind enum class
    py::enum_<plssvm::backend_type>(m, "backend_type")
        .value("automatic", plssvm::backend_type::automatic)
        .value("openmp", plssvm::backend_type::openmp)
        .value("cuda", plssvm::backend_type::cuda)
        .value("hip", plssvm::backend_type::hip)
        .value("opencl", plssvm::backend_type::opencl)
        .value("sycl", plssvm::backend_type::sycl);

    // bind free functions
    m.def("list_available_backends", &plssvm::list_available_backends);
    m.def("determine_default_backend", &plssvm::determine_default_backend,
          py::arg("available_backends") = plssvm::list_available_backends(),
          py::arg("available_target_platforms") = plssvm::list_available_target_platforms());
}