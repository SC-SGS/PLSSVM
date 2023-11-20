/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "utility.hpp"

#include "fmt/core.h"            // fmt::format
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self
#include "pybind11/stl.h"        // support for STL types

namespace py = pybind11;

void init_parameter(py::module_ &m) {
    // bind parameter class
    py::class_<plssvm::parameter>(m, "Parameter")
        .def(py::init<>())
        .def(py::init<plssvm::kernel_function_type, int, plssvm::real_type, plssvm::real_type, plssvm::real_type>())
        .def(py::init([](const py::kwargs &args) {
                 // check for valid keys
                 check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                 // if one of the value named parameter is provided, set the respective value
                 return convert_kwargs_to_parameter(args);
             }),
             "create a new SVM parameter object")
        .def_property(
            "kernel_type",
            [](const plssvm::parameter &self) { return self.kernel_type.value(); },
            [](plssvm::parameter &self, const plssvm::kernel_function_type kernel_type) { self.kernel_type = kernel_type; },
            py::return_value_policy::reference,
            "change the used kernel function: linear, polynomial, and rbf")
        .def_property(
            "degree",
            [](const plssvm::parameter &self) { return self.degree.value(); },
            [](plssvm::parameter &self, const int degree) { self.degree = degree; },
            py::return_value_policy::reference,
            "change the degree parameter for the polynomial kernel function")
        .def_property(
            "gamma",
            [](const plssvm::parameter &self) { return self.gamma.value(); },
            [](plssvm::parameter &self, const plssvm::real_type gamma) { self.gamma = gamma; },
            py::return_value_policy::reference,
            "change the gamma parameter for the polynomial and rbf kernel functions")
        .def_property(
            "coef0",
            [](const plssvm::parameter &self) { return self.coef0.value(); },
            [](plssvm::parameter &self, const plssvm::real_type coef0) { self.coef0 = coef0; },
            py::return_value_policy::reference,
            "change the coef0 parameter for the polynomial kernel function")
        .def_property(
            "cost",
            [](const plssvm::parameter &self) { return self.cost.value(); },
            [](plssvm::parameter &self, const plssvm::real_type cost) { self.cost = cost; },
            py::return_value_policy::reference,
            "change the cost parameter for the CSVM")
        .def("equivalent", &plssvm::parameter::equivalent, "check whether two parameter objects are equivalent, i.e., the SVM parameter important for the current 'kernel_type' are the same")
        .def(py::self == py::self, "check whether two parameter objects are identical")
        .def(py::self != py::self, "check whether two parameter objects are different")
        .def("__repr__", [](const plssvm::parameter &self) {
            return fmt::format("<plssvm.Parameter with {{ kernel_type: {}, degree: {}, gamma:{}, coef0: {}, cost: {} }}>",
                               self.kernel_type,
                               self.degree,
                               self.gamma,
                               self.coef0,
                               self.cost);
        });

    // bind free functions
    m.def("equivalent", &plssvm::equivalent, "check whether two parameter objects are equivalent, i.e., the SVM parameter important for the current 'kernel_type' are the same");
}