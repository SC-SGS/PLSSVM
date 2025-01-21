/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"            // plssvm::hpx::backend_csvm_type_t
#include "plssvm/backends/HPX/csvm.hpp"        // plssvm::hpx::csvm
#include "plssvm/backends/HPX/exceptions.hpp"  // plssvm::hpx::backend_exception
#include "plssvm/exceptions/exceptions.hpp"    // plssvm::exception
#include "plssvm/parameter.hpp"                // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                 // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                 // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                 // plssvm::csvr
#include "plssvm/target_platforms.hpp"         // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kwargs, py::exception
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique
#include <string>  // std::string

namespace py = pybind11;

template <typename csvm_type>
void bind_hpx_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::hpx::backend_csvm_type_t<csvm_type>;

    py::class_<backend_csvm_type, plssvm::hpx::csvm, csvm_type>(m, csvm_name.c_str())
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

void init_hpx_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the HPX C-SVM bindings
    py::module_ hpx_module = m.def_submodule("hpx", "a module containing all HPX backend specific functionality");
    const py::module_ hpx_pure_virtual_module = hpx_module.def_submodule("__pure_virtual", "a module containing all pure-virtual HPX backend specific functionality");

    // bind the pure-virtual base HPX C-SVM
    py::class_<plssvm::hpx::csvm, plssvm::csvm>(hpx_pure_virtual_module, "__pure_virtual_hpx_base_CSVM");

    // bind the specific HPX C-SVC and C-SVR classes
    bind_hpx_csvms<plssvm::csvc>(hpx_module, "CSVC");
    bind_hpx_csvms<plssvm::csvr>(hpx_module, "CSVR");

    // register HPX backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::hpx::backend_exception>(hpx_module, "BackendError", base_exception);
}
