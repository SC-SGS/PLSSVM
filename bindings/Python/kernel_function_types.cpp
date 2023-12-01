/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_function_types.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "pybind11/pybind11.h"   // py::module_, py::enum_, py::arg, py::pos_only
#include "pybind11/stl.h"        // support for STL types: std::vector, std::optional

#include <optional>              // std::optional, std::nullopt
#include <vector>                // std::vector

namespace py = pybind11;

void init_kernel_function_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::kernel_function_type>(m, "KernelFunctionType")
        .value("LINEAR", plssvm::kernel_function_type::linear, "linear kernel function: <u, v>")
        .value("POLYNOMIAL", plssvm::kernel_function_type::polynomial, "polynomial kernel function: (gamma * <u, v> + coef0)^degree")
        .value("RBF", plssvm::kernel_function_type::rbf, "radial basis function: e^(-gamma * ||u - v||^2)");

    // bind free functions
    m.def("kernel_function_type_to_math_string", &plssvm::kernel_function_type_to_math_string, "return the mathematical representation of a KernelFunctionType");

    const plssvm::parameter default_params{};

    m.def("linear_kernel_function", &plssvm::kernel_function<plssvm::kernel_function_type::linear, plssvm::real_type>, "apply the linear kernel function to two vectors");
    m.def(
        "polynomial_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const int degree, const std::optional<plssvm::real_type> gamma, const plssvm::real_type coef0) {
            return plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x, y, degree, gamma.has_value() ? gamma.value() : plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()), coef0);
        },
        "apply the polynomial kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("degree") = default_params.degree.value(),
        py::arg("gamma") = std::nullopt,
        py::arg("coef0") = default_params.coef0.value());
    m.def(
        "rbf_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const std::optional<plssvm::real_type> gamma) {
            return plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x, y, gamma.has_value() ? gamma.value() : plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
        },
        "apply the radial basis function kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("gamma") = std::nullopt);

    m.def(
        "kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, plssvm::parameter params) {
            // set default gamma value
            if (params.gamma.is_default()) {
                params.gamma = plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size());
            }
            return plssvm::kernel_function(x, y, params);
        },
        "apply the kernel function defined in the parameter object to two vectors");
}
