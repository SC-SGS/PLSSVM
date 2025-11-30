/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm/csvm.hpp"  // plssvm::csvm

#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/gamma.hpp"
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter, named arguments

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings
#include "bindings/Python/utility.hpp"       // plssvm::bindings::python::util::check_kwargs_for_correctness

#include "pybind11/cast.h"      // py::arg
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::arg, py::kwargs, py::module_local
#include "pybind11/pytypes.h"   // py::kwargs
#include "pybind11/stl.h"       // NOLINT: support for STL types: std::variant

namespace py = pybind11;

void init_csvm(py::module_ &m) {
    py::class_<plssvm::csvm>(m, "CSVM", py::module_local(), "Base class for all other C-SVC or C-SVR implementations.")
        .def("get_params", &plssvm::csvm::get_params, py::return_value_policy::copy, "get the hyper-parameters used for this C-SVM")
        .def("set_params", [](plssvm::csvm &self, const plssvm::parameter &params) { self.set_params(params); }, "update the hyper-parameters used for this C-SVM using a plssvm.Parameter object", py::arg("params"))
        .def("set_params", [](plssvm::csvm &self, const py::kwargs &args) {
                // check keyword arguments
                plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                // convert kwargs to parameter and update csvm internal parameter
                if (args.contains("kernel_type")) {
                    self.set_params(plssvm::kernel_type = args["kernel_type"].cast<plssvm::kernel_function_type>());
                }
                if (args.contains("degree")) {
                    self.set_params(plssvm::degree = args["degree"].cast<int>());
                }
                if (args.contains("gamma")) {
                    self.set_params(plssvm::gamma = args["gamma"].cast<plssvm::gamma_type>());
                }
                if (args.contains("coef0")) {
                    self.set_params(plssvm::coef0 = args["coef0"].cast<plssvm::real_type>());
                }
                if (args.contains("cost")) {
                    self.set_params(plssvm::cost = args["cost"].cast<plssvm::real_type>());
                } }, "update the hyper-parameters used for this C-SVM using keyword arguments")
        .def("get_target_platform", &plssvm::csvm::get_target_platform, "get the actual target platform this C-SVM runs on")
        .def("num_available_devices", &plssvm::csvm::num_available_devices, "get the number of available devices for the current C-SVM")
        .def("communicator", &plssvm::csvm::communicator, "the associated MPI communicator");
}
