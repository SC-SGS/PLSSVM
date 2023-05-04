/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/csvm.hpp"
#include "plssvm/backends/OpenMP/exceptions.hpp"

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "../utility.hpp"               // check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception

#include "pybind11/pybind11.h"          // py::module_, py::class_, py::init
#include "pybind11/stl.h"               // support for STL types

#include <memory>                       // std::make_unique

namespace py = pybind11;

void init_openmp_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the OpenMP CSVM bindings
    py::module_ openmp_module = m.def_submodule("openmp", "a module containing all OpenMP backend specific functionality");

    // bind the CSVM using the OpenMP backend
    py::class_<plssvm::openmp::csvm, plssvm::csvm>(openmp_module, "CSVM")
        .def(py::init<>(), "create an SVM with the automatic target platform and default parameter object")
        .def(py::init<plssvm::parameter>(), "create an SVM with the automatic target platform and provided parameter object")
        .def(py::init<plssvm::target_platform>(), "create an SVM with the provided target platform and default parameter object")
        .def(py::init<plssvm::target_platform, plssvm::parameter>(), "create an SVM with the provided target platform and parameter object")
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the default target platform
                 return std::make_unique<plssvm::openmp::csvm>(params);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the provided target platform
                 return std::make_unique<plssvm::openmp::csvm>(target, params);
             }),
             "create an SVM with the provided target platform and keyword arguments");

    // register OpenMP backend specific exceptions
    register_py_exception<plssvm::openmp::backend_exception>(openmp_module, "BackendError", base_exception);
}