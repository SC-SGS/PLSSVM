/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"                         // plssvm::stdpar::backend_csvm_type_t
#include "plssvm/backends/stdpar/csvm.hpp"                  // plssvm::stdpar::csvm
#include "plssvm/backends/stdpar/exceptions.hpp"            // plssvm::stdpar::backend_exception
#include "plssvm/backends/stdpar/implementation_types.hpp"  // plssvm::stdpar::implementation_type
#include "plssvm/exceptions/exceptions.hpp"                 // plssvm::exception
#include "plssvm/parameter.hpp"                             // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                              // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                              // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                              // plssvm::csvr
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kwargs, py::exception
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique
#include <string>  // std::string

namespace py = pybind11;

template <typename csvm_type>
void bind_stdpar_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::stdpar::backend_csvm_type_t<csvm_type>;

    py::class_<backend_csvm_type, plssvm::stdpar::csvm, csvm_type>(m, csvm_name.c_str())
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the provided target platform
                 return std::make_unique<backend_csvm_type>(target, params);
             }),
             "create an SVM with the provided target platform and keyword arguments");
}

void init_stdpar_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the stdpar C-SVM bindings
    py::module_ stdpar_module = m.def_submodule("stdpar", "a module containing all stdpar backend specific functionality");
    const py::module_ stdpar_pure_virtual_module = stdpar_module.def_submodule("__pure_virtual", "a module containing all pure-virtual stdpar backend specific functionality");

    // bind the enum class
    py::enum_<plssvm::stdpar::implementation_type>(stdpar_module, "ImplementationType")
        .value("NVHPC", plssvm::stdpar::implementation_type::nvhpc, "use NVIDIA's HPC SDK (NVHPC) compiler nvc++")
        .value("ROC_STDPAR", plssvm::stdpar::implementation_type::roc_stdpar, "use AMD's roc-stdpar compiler (patched LLVM)")
        .value("INTEL_LLVM", plssvm::stdpar::implementation_type::intel_llvm, "use Intel's LLVM compiler icpx")
        .value("ADAPTIVECPP", plssvm::stdpar::implementation_type::adaptivecpp, "use AdaptiveCpp (formerly known as hipSYCL)")
        .value("GNU_TBB", plssvm::stdpar::implementation_type::gnu_tbb, "use GNU GCC + Intel's TBB library");

    stdpar_module.def("list_available_stdpar_implementations", &plssvm::stdpar::list_available_stdpar_implementations, "list all available stdpar implementations");

    // bind the pure-virtual base stdpar C-SVM
    py::class_<plssvm::stdpar::csvm, plssvm::csvm>(stdpar_pure_virtual_module, "__pure_virtual_stdpar_base_CSVM");

    // bind the specific stdpar C-SVC and C-SVR classes
    bind_stdpar_csvms<plssvm::csvc>(stdpar_module, "CSVC");
    bind_stdpar_csvms<plssvm::csvr>(stdpar_module, "CSVR");

    // register stdpar backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::stdpar::backend_exception>(stdpar_module, "BackendError", base_exception);
}
