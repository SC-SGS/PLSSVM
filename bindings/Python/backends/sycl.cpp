#include "plssvm/backends/SYCL/exceptions.hpp"
#include "plssvm/backends/SYCL/implementation_type.hpp"
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"

#include "../utility.hpp"  // PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module, py::enum_
#include "pybind11/stl.h"       // support for STL types

namespace py = pybind11;

void init_sycl(py::module &m) {
    // use its own submodule for the DPCPP CSVM bindings
    py::module sycl_module = m.def_submodule("sycl");

    // register SYCL backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::sycl::backend_exception, sycl_module, backend_error)

    // bind the two enum classes
    py::enum_<plssvm::sycl::implementation_type>(sycl_module, "implementation_type")
        .value("automatic", plssvm::sycl::implementation_type::automatic)
        .value("dpcpp", plssvm::sycl::implementation_type::dpcpp)
        .value("hipsycl", plssvm::sycl::implementation_type::hipsycl);

    py::enum_<plssvm::sycl::kernel_invocation_type>(sycl_module, "kernel_invocation_type")
        .value("automatic", plssvm::sycl::kernel_invocation_type::automatic)
        .value("nd_range", plssvm::sycl::kernel_invocation_type::nd_range)
        .value("hierarchical", plssvm::sycl::kernel_invocation_type::hierarchical);
}