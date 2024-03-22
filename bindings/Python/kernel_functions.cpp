/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_functions.hpp"  // plssvm::kernel_function

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "pybind11/pybind11.h"  // py::module_, py::arg, py::pos_only
#include "pybind11/stl.h"       // support for STL types: std::vector, std::optional

#include <optional>  // std::optional, std::nullopt
#include <vector>    // std::vector

namespace py = pybind11;

void init_kernel_functions(py::module_ &m) {
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
        "sigmoid_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const std::optional<plssvm::real_type> gamma, const plssvm::real_type coef0) {
            return plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(x, y, gamma.has_value() ? gamma.value() : plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()), coef0);
        },
        "apply the sigmoid kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("gamma") = std::nullopt,
        py::arg("coef0") = default_params.coef0.value());
    m.def(
        "laplacian_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const std::optional<plssvm::real_type> gamma) {
            return plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(x, y, gamma.has_value() ? gamma.value() : plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
        },
        "apply the laplacian kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::pos_only(),
        py::arg("gamma") = std::nullopt);
    m.def(
        "chi_squared_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const std::optional<plssvm::real_type> gamma) {
            return plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(x, y, gamma.has_value() ? gamma.value() : plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
        },
        "apply the chi-squared kernel function to two vectors",
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
