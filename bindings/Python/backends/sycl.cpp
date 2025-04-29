/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/exceptions.hpp"               // plssvm::sycl::backend_exception
#include "plssvm/backends/SYCL/implementation_types.hpp"     // plssvm::sycl::{implementation_type, list_available_sycl_implementations}
#include "plssvm/backends/SYCL/kernel_invocation_types.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/exceptions/exceptions.hpp"                  // plssvm::exception

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{register_py_exception, register_implicit_str_enum_conversion}

#include "pybind11/pybind11.h"  // py::module_, py::enum_, py::exception
#include "pybind11/stl.h"       // support for STL types: std:vector

#include <vector>  // std::vector

#define PLSSVM_CONCATENATE_DETAIL(x, y) x##y
#define PLSSVM_CONCATENATE(x, y) PLSSVM_CONCATENATE_DETAIL(x, y)

namespace py = pybind11;

py::module_ init_adaptivecpp_csvm(py::module_ &, const py::exception<plssvm::exception> &);
py::module_ init_dpcpp_csvm(py::module_ &, const py::exception<plssvm::exception> &);

void init_sycl(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the SYCL specific bindings
    py::module_ sycl_module = m.def_submodule("sycl", "a module containing all SYCL backend specific functionality");

    // register SYCL backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::sycl::backend_exception>(sycl_module, "BackendError", base_exception);

    // bind the two enum classes
    py::enum_<plssvm::sycl::implementation_type> py_enum_impl(sycl_module, "ImplementationType", "Enum class for all supported SYCL implementation in PLSSVM.");
    py_enum_impl
        .value("AUTOMATIC", plssvm::sycl::implementation_type::automatic, "use the available SYCL implementation; if more than one implementation is available, the macro PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION must be defined during the CMake configuration")
        .value("DPCPP", plssvm::sycl::implementation_type::dpcpp, "use DPC++ as SYCL implementation")
        .value("ADAPTIVECPP", plssvm::sycl::implementation_type::adaptivecpp, "use AdaptiveCpp (formerly known as hipSYCL) as SYCL implementation");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::sycl::implementation_type>(py_enum_impl);

    sycl_module.def("list_available_sycl_implementations", &plssvm::sycl::list_available_sycl_implementations, "list all available SYCL implementations");

    py::enum_<plssvm::sycl::kernel_invocation_type> py_enum_invocation(sycl_module, "KernelInvocationType", "Enum class for all possible SYCL kernel invocation types supported in PLSSVM.");
    py_enum_invocation
        .value("AUTOMATIC", plssvm::sycl::kernel_invocation_type::automatic, "use the best kernel invocation type for the current SYCL implementation and target hardware platform")
        .value("ND_RANGE", plssvm::sycl::kernel_invocation_type::nd_range, "use the nd_range kernel invocation type");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::sycl::kernel_invocation_type>(py_enum_invocation);

    // initialize SYCL binding classes
#if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    const py::module_ adaptivecpp_module = init_adaptivecpp_csvm(m, base_exception);
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    const py::module_ dpcpp_module = init_dpcpp_csvm(m, base_exception);
#endif

    // "alias" one of the DPC++ or AdaptiveCpp C-SVCs and C-SVRs to be the respective default SYCL C-SVC and C-SVR
    sycl_module.attr("CSVC") = PLSSVM_CONCATENATE(PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION, _module).attr("CSVC");
    sycl_module.attr("CSVR") = PLSSVM_CONCATENATE(PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION, _module).attr("CSVR");
}
