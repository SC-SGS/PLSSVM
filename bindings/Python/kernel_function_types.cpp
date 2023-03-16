#include "plssvm/kernel_function_types.hpp"
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "fmt/core.h"           // fmt::format
#include "pybind11/pybind11.h"  // py::module, py::enum_, py::value_error
#include "pybind11/stl.h"       // support for STL types

#include <optional>  // std::optional, std::nullopt
#include <vector>    // std::vector

namespace py = pybind11;

void init_kernel_function_types(py::module &m) {
    // TODO: performance + types (float)?
    // bind enum class
    py::enum_<plssvm::kernel_function_type>(m, "kernel_function_type")
        .value("linear", plssvm::kernel_function_type::linear)
        .value("polynomial", plssvm::kernel_function_type::polynomial)
        .value("rbf", plssvm::kernel_function_type::rbf);

    // bind free functions
    m.def("kernel_function_type_to_math_string", &plssvm::kernel_function_type_to_math_string);

    const plssvm::parameter params{};

    m.def("linear_kernel_function", &plssvm::kernel_function<plssvm::kernel_function_type::linear, double>);
    m.def(
        "polynomial_kernel_function", [](const std::vector<double> &x, const std::vector<double> &y, const int degree, const std::optional<double> gamma, const double coef0) {
            return plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x, y, degree, gamma.has_value() ? gamma.value() : 1.0 / x.size(), coef0);
        },
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("degree") = params.degree.value(),
        py::arg("gamma") = std::nullopt,
        py::arg("coef0") = params.coef0.value());
    m.def(
        "rbf_kernel_function", [](const std::vector<double> &x, const std::vector<double> &y, const std::optional<double> gamma) {
            return plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x, y, gamma.has_value() ? gamma.value() : 1.0 / x.size());
        },
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("gamma") = std::nullopt);
}
