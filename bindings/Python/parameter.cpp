/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter.hpp"  // plssvm::parameter, plssvm::equivalent

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/gamma.hpp"                  // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings

#include "fmt/format.h"          // fmt::format
#include "pybind11/cast.h"       // py::make_tuple
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::self, py::pickle, py::return_value_policy
#include "pybind11/pytypes.h"    // py::tuple
#include "pybind11/stl.h"        // NOLINT: support for STL types

#include <cstddef>    // std::size_t
#include <stdexcept>  // std::runtime_error

namespace py = pybind11;

void init_parameter(py::module_ &m) {
    const plssvm::parameter default_params{};

    // bind parameter class
    py::class_<plssvm::parameter>(m, "Parameter", "A class for encapsulating all important C-SVM hyper-parameters.")
        .def(py::init([](const plssvm::kernel_function_type kernel_type, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type cost) {
                 return plssvm::parameter{ kernel_type, degree, gamma, coef0, cost };
             }),
             "create a new Parameter object with the optionally provided hyper-parameter values",
             py::arg("kernel_type") = default_params.kernel_type,
             py::arg("degree") = default_params.degree,
             py::arg("gamma") = default_params.gamma,
             py::arg("coef0") = default_params.coef0,
             py::arg("cost") = default_params.cost)
        .def_property(
            "kernel_type",
            [](const plssvm::parameter &self) { return self.kernel_type; },
            [](plssvm::parameter &self, const plssvm::kernel_function_type kernel_type) { self.kernel_type = kernel_type; },
            py::return_value_policy::reference,
            "change the used kernel function: linear, polynomial, rbf, sigmoid, laplacian, or chi_squared")
        .def_property(
            "degree",
            [](const plssvm::parameter &self) { return self.degree; },
            [](plssvm::parameter &self, const int degree) { self.degree = degree; },
            py::return_value_policy::reference,
            "change the degree parameter for the polynomial kernel function")
        .def_property(
            "gamma",
            [](const plssvm::parameter &self) { return self.gamma; },
            [](plssvm::parameter &self, const plssvm::gamma_type &gamma) { self.gamma = gamma; },
            py::return_value_policy::reference,
            "change the gamma parameter for all kernel functions except the linear one")
        .def_property(
            "coef0",
            [](const plssvm::parameter &self) { return self.coef0; },
            [](plssvm::parameter &self, const plssvm::real_type coef0) { self.coef0 = coef0; },
            py::return_value_policy::reference,
            "change the coef0 parameter for the polynomial and sigmoid kernel functions")
        .def_property(
            "cost",
            [](const plssvm::parameter &self) { return self.cost; },
            [](plssvm::parameter &self, const plssvm::real_type cost) { self.cost = cost; },
            py::return_value_policy::reference,
            "change the cost parameter for the C-SVM")
        .def("equivalent", &plssvm::parameter::equivalent, "check whether two parameter objects are equivalent, i.e., the SVM hyper-parameters important for the current 'kernel_type' are the same")
        .def(py::self == py::self, "check whether two parameter objects are identical")
        .def(py::self != py::self, "check whether two parameter objects are different")
        .def("__repr__", [](const plssvm::parameter &self) {
            return fmt::format("<plssvm.Parameter with {{ kernel_type: {}, degree: {}, gamma: {}, coef0: {}, cost: {} }}>",
                               self.kernel_type,
                               self.degree,
                               self.gamma,
                               self.coef0,
                               self.cost);
        })
        .def(py::pickle(
            // clang-format off
            [](const plssvm::parameter &self) {  // __getstate__
                // return a tuple that fully encodes the state of the object
                return py::make_tuple(self.kernel_type, self.degree, self.gamma, self.coef0, self.cost);
            },
            [](py::tuple t) {  // __setstate__
                if (t.size() != 5) {
                    throw std::runtime_error{ "Invalid state!" };
                }
                // create a new C++ instance
                return plssvm::parameter{ t[0].cast<plssvm::kernel_function_type>(), t[1].cast<int>(), t[2].cast<plssvm::real_type>(), t[3].cast<plssvm::real_type>(), t[4].cast<plssvm::real_type>() };
            }
            )
             // clang-format on
        );

    // bind free functions
    m.def("equivalent", &plssvm::equivalent, "check whether two parameter objects are equivalent, i.e., the SVM hyper-parameters important for the current 'kernel_type' are the same");
}
