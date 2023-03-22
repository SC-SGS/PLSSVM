#include "plssvm/backends/SYCL/exceptions.hpp"
#include "plssvm/backends/SYCL/implementation_type.hpp"
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"

#include "../utility.hpp"  // PLSSVM_REGISTER_EXCEPTION

#include "pybind11/pybind11.h"  // py::module_, py::enum_, py::exception
#include "pybind11/stl.h"       // support for STL types: std:vector

namespace py = pybind11;

void init_sycl(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the SYCL specific bindings
    py::module_ sycl_module = m.def_submodule("sycl", "a module containing all SYCL backend specific functionality");

    // register SYCL backend specific exceptions
    PLSSVM_REGISTER_EXCEPTION(plssvm::sycl::backend_exception, sycl_module, BackendError, base_exception)

    // bind the two enum classes
    py::enum_<plssvm::sycl::implementation_type>(sycl_module, "ImplementationType")
        .value("AUTOMATIC", plssvm::sycl::implementation_type::automatic, "use the available SYCL implementation; if more than one implementation is available, the macro PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION must be defined during the CMake configuration")
        .value("DPCPP", plssvm::sycl::implementation_type::dpcpp, "use DPC++ as SYCL implementation")
        .value("HIPSYCL", plssvm::sycl::implementation_type::hipsycl, "use hipSYCL as SYCL implementation");

    sycl_module.def("list_available_sycl_implementations", &plssvm::sycl::list_available_sycl_implementations, "list all available SYCL implementations");

    py::enum_<plssvm::sycl::kernel_invocation_type>(sycl_module, "KernelInvocationType")
        .value("AUTOMATIC", plssvm::sycl::kernel_invocation_type::automatic, "use the best kernel invocation type for the current SYCL implementation and target hardware platform")
        .value("ND_RANGE", plssvm::sycl::kernel_invocation_type::nd_range, "use the nd_range kernel invocation type")
        .value("HIERARCHICAL", plssvm::sycl::kernel_invocation_type::hierarchical, "use the hierarchical kernel invocation type");
}