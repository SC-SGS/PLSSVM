/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm/csvm.hpp"  // plssvm::csvm

#include "plssvm/parameter.hpp"  // plssvm::parameter, named parameters

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, convert_kwargs_to_parameter}

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs

namespace py = pybind11;

void init_csvm(py::module_ &pure_virtual) {
    py::class_<plssvm::csvm>(pure_virtual, "__pure_virtual_base_CSVM", "Base class for all other C-SVC or C-SVR implementations.")
        .def("get_params", &plssvm::csvm::get_params, "get the hyper-parameters used for this C-SVM")
        .def("set_params", [](plssvm::csvm &self, const plssvm::parameter &params) { self.set_params(params); }, "update the hyper-parameters used for this C-SVM using a plssvm.Parameter object")
        .def("set_params", [](plssvm::csvm &self, const py::kwargs &args) {
                // check keyword arguments
                plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                // convert kwargs to parameter and update csvm internal parameter
                self.set_params(plssvm::bindings::python::util::convert_kwargs_to_parameter(args, self.get_params())); }, "update the hyper-parameters used for this C-SVM using keyword arguments")
        .def("get_target_platform", &plssvm::csvm::get_target_platform, "get the actual target platform this C-SVM runs on")
        .def("num_available_devices", &plssvm::csvm::num_available_devices, "get the number of available devices for the current C-SVM")
        .def("communicator", &plssvm::csvm::communicator, "the associated MPI communicator");
}
