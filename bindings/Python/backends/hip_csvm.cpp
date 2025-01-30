/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"            // plssvm::hip::backend_csvm_type_t
#include "plssvm/backends/HIP/csvm.hpp"        // plssvm::hip::csvm
#include "plssvm/backends/HIP/exceptions.hpp"  // plssvm::hip::backend_exception
#include "plssvm/exceptions/exceptions.hpp"    // plssvm::exception
#include "plssvm/parameter.hpp"                // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                 // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                 // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                 // plssvm::csvr
#include "plssvm/target_platforms.hpp"         // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::exception
#include "pybind11/pytypes.h"   // py::kwargs

#include <memory>  // std::make_unique
#include <string>  // std::string

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_hip_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::hip::backend_csvm_type_t<csvm_type>;

    // assemble docstrings
    const std::string param_docstring{ fmt::format("create a HIP {} with the provided parameters", csvm_name) };
    const std::string target_param_docstring{ fmt::format("create a HIP {} with the provided target platform and parameters", csvm_name) };
    const std::string kwargs_docstring{ fmt::format("create a HIP {} with the provided keyword arguments", csvm_name) };
    const std::string target_kwargs_docstring{ fmt::format("create a HIP {} with the provided target platform and keyword arguments", csvm_name) };

    py::class_<backend_csvm_type, plssvm::hip::csvm, csvm_type>(m, csvm_name.c_str())
        .def(py::init<plssvm::parameter>(), param_docstring.c_str())
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), target_param_docstring.c_str())
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params);
             }),
             kwargs_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // create C-SVM with the provided target platform
                 return std::make_unique<backend_csvm_type>(target, params);
             }),
             target_kwargs_docstring.c_str());
}

}  // namespace

void init_hip_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the HIP C-SVM bindings
    py::module_ hip_module = m.def_submodule("hip", "a module containing all HIP backend specific functionality");
    const py::module_ hip_pure_virtual_module = hip_module.def_submodule("__pure_virtual", "a module containing all pure-virtual HIP backend specific functionality");

    // bind the pure-virtual base HIP C-SVM
    [[maybe_unused]] const py::class_<plssvm::hip::csvm, plssvm::csvm> virtual_base_hip_csvm(hip_pure_virtual_module, "__pure_virtual_hip_base_CSVM");

    // bind the specific HIP C-SVC and C-SVR classes
    bind_hip_csvms<plssvm::csvc>(hip_module, "CSVC");
    bind_hip_csvms<plssvm::csvr>(hip_module, "CSVR");

    // register HIP backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::hip::backend_exception>(hip_module, "BackendError", base_exception);
}
