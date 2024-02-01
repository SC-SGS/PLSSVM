/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/csvm.hpp"        // plssvm::opencl::csvm
#include "plssvm/backends/OpenCL/exceptions.hpp"  // plssvm::opencl::backend_exception
#include "plssvm/csvm.hpp"                        // plssvm::csvm
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter, register_py_exception

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init
#include "pybind11/stl.h"       // support for STL types

#include <memory>  // std::make_unique

namespace py = pybind11;

void init_opencl_csvm(py::module_ &m, const py::exception<plssvm::exception> &base_exception) {
    // use its own submodule for the OpenCL CSVM bindings
    py::module_ opencl_module = m.def_submodule("opencl", "a module containing all OpenCL backend specific functionality");

    // bind the CSVM using the OpenCL backend
    py::class_<plssvm::opencl::csvm, plssvm::csvm>(opencl_module, "CSVM")
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
                 return std::make_unique<plssvm::opencl::csvm>(params);
             }),
             "create an SVM with the default target platform and keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value keyword parameter is provided, set the respective value
                 const plssvm::parameter params = convert_kwargs_to_parameter(args);
                 // create CSVM with the provided target platform
                 return std::make_unique<plssvm::opencl::csvm>(target, params);
             }),
             "create an SVM with the provided target platform and keyword arguments")
        .def("num_available_devices", &plssvm::opencl::csvm::num_available_devices, "the number of available devices");

    // register OpenCL backend specific exceptions
    register_py_exception<plssvm::opencl::backend_exception>(opencl_module, "BackendError", base_exception);
}
