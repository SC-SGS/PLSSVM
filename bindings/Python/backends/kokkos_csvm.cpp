/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backend_types.hpp"                    // plssvm::kokkos::backend_csvm_type_t
#include "plssvm/backends/Kokkos/csvm.hpp"             // plssvm::kokkos::csvm
#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::exception
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                         // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                         // plssvm::csvm
#include "plssvm/svm/csvr.hpp"                         // plssvm::csvr
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::exception
#include "pybind11/pytypes.h"   // py::kwargs

#include <memory>  // std::make_unique
#include <string>  // std::string

namespace py = pybind11;

namespace {

template <typename csvm_type>
void bind_kokkos_csvms(py::module_ &m, const std::string &csvm_name) {
    using backend_csvm_type = plssvm::kokkos::backend_csvm_type_t<csvm_type>;

    // assemble docstrings
    const std::string param_docstring{ fmt::format("create a Kokkos {} with the provided parameters and optional Kokkos specific keyword arguments", csvm_name) };
    const std::string target_param_docstring{ fmt::format("create a Kokkos {} with the provided target platform, parameters, and optional Kokkos specific keyword arguments", csvm_name) };
    const std::string kwargs_docstring{ fmt::format("create a Kokkos {} with the provided keyword arguments (including optional Kokkos specific keyword arguments)", csvm_name) };
    const std::string target_kwargs_docstring{ fmt::format("create a Kokkos {} with the provided target platform and keyword arguments (including optional Kokkos specific keyword arguments)", csvm_name) };

    py::class_<backend_csvm_type, plssvm::kokkos::csvm, csvm_type>(m, csvm_name.c_str())
        .def(py::init([](const plssvm::parameter params, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kokkos_execution_space" });
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params, plssvm::kokkos_execution_space = space);
             }),
             param_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter params, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kokkos_execution_space" });
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(target, params, plssvm::kokkos_execution_space = space);
             }),
             target_param_docstring.c_str())
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "kokkos_execution_space" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create C-SVM with the default target platform
                 return std::make_unique<backend_csvm_type>(params, plssvm::kokkos_execution_space = space);
             }),
             kwargs_docstring.c_str())
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "kokkos_execution_space" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = plssvm::bindings::python::util::convert_kwargs_to_parameter(args);
                 // set Kokkos execution space
                 const plssvm::kokkos::execution_space space = args.contains("kokkos_execution_space") ? args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>() : plssvm::kokkos::execution_space::automatic;
                 // create C-SVM with the provided target platform
                 return std::make_unique<backend_csvm_type>(target, params, plssvm::kokkos_execution_space = space);
             }),
             target_kwargs_docstring.c_str())
        .def("get_execution_space", &plssvm::kokkos::csvm::get_execution_space, "get the Kokkos execution space used in this Kokkos C-SVM");
}

}  // namespace

void init_kokkos_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the Kokkos C-SVM bindings
    py::module_ kokkos_module = m.def_submodule("kokkos", "a module containing all Kokkos backend specific functionality");
    const py::module_ kokkos_pure_virtual_module = kokkos_module.def_submodule("__pure_virtual", "a module containing all pure-virtual Kokkos backend specific functionality");

    // bind the pure-virtual base Kokkos C-SVM
    [[maybe_unused]] const py::class_<plssvm::kokkos::csvm, plssvm::csvm> virtual_base_kokkos_csvm(kokkos_pure_virtual_module, "__pure_virtual_kokkos_base_CSVM");

    // bind the specific Kokkos C-SVC and C-SVR classes
    bind_kokkos_csvms<plssvm::csvc>(kokkos_module, "CSVC");
    bind_kokkos_csvms<plssvm::csvr>(kokkos_module, "CSVR");

    // register Kokkos backend specific exceptions
    plssvm::bindings::python::util::register_py_exception<plssvm::kokkos::backend_exception>(kokkos_module, "BackendError", base_exception);
}
