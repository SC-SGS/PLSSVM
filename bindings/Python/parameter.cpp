#include "plssvm/parameter.hpp"

#include "fmt/core.h"            // fmt::format
#include "pybind11/operators.h"  // support for operators
#include "pybind11/pybind11.h"   // py::module_, py::class_, py::init, py::arg, py::return_value_policy, py::self
#include "pybind11/stl.h"        // support for STL types

#include <sstream>

namespace py = pybind11;

void init_parameter(py::module_ &m) {
    const plssvm::parameter default_params{};

    // bind parameter class
    py::class_<plssvm::parameter>(m, "parameter")
        .def(py::init<>())
        .def(py::init<plssvm::kernel_function_type, int, double, double, double>(),
             py::arg("kernel_type") = default_params.kernel_type.value(),
             py::arg("degree") = default_params.degree.value(),
             py::arg("gamma") = default_params.gamma.value(),
             py::arg("coef0") = default_params.coef0.value(),
             py::arg("cost") = default_params.cost.value())
        .def_property(
            "kernel_type",
            [](const plssvm::parameter &param) { return param.kernel_type.value(); },
            [](plssvm::parameter &param, const plssvm::kernel_function_type kernel_type) { param.kernel_type = kernel_type; },
            py::return_value_policy::reference)
        .def_property(
            "degree",
            [](const plssvm::parameter &param) { return param.degree.value(); },
            [](plssvm::parameter &param, const int degree) { param.degree = degree; },
            py::return_value_policy::reference)
        .def_property(
            "degree",
            [](const plssvm::parameter &param) { return param.gamma.value(); },
            [](plssvm::parameter &param, const double gamma) { param.gamma = gamma; },
            py::return_value_policy::reference)
        .def_property(
            "degree",
            [](const plssvm::parameter &param) { return param.coef0.value(); },
            [](plssvm::parameter &param, const double coef0) { param.coef0 = coef0; },
            py::return_value_policy::reference)
        .def_property(
            "degree",
            [](const plssvm::parameter &param) { return param.cost.value(); },
            [](plssvm::parameter &param, const double cost) { param.cost = cost; },
            py::return_value_policy::reference)
        .def("equivalent", &plssvm::parameter::equivalent)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("__repr__", [](const plssvm::parameter &params) {
            return fmt::format("<plssvm.parameter with {{ kernel_type: {}, degree: {}, gamma:{}, coef0: {}, cost: {} }}>",
                               params.kernel_type, params.degree, params.gamma, params.coef0, params.cost);
        });

    // bind free functions
    m.def("equivalent", &plssvm::detail::equivalent<double>);
}