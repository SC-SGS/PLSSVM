/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/kernel_functions.hpp"  // plssvm::kernel_function

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/gamma.hpp"                  // plssvm::gamma_coefficient_type, plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "pybind11/pybind11.h"  // py::module_, py::arg, py::kw_only
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <variant>  // std::holds_alternative, std::get
#include <vector>   // std::vector

namespace py = pybind11;

void init_kernel_functions(py::module_ &m) {
    const plssvm::parameter default_params{};

    m.def("linear_kernel_function", &plssvm::kernel_function<plssvm::kernel_function_type::linear, plssvm::real_type>, "apply the linear kernel function to two vectors", py::arg("x"), py::arg("y"));
    m.def(
        "polynomial_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0) {
            if (std::holds_alternative<plssvm::real_type>(gamma)) {
                return plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x, y, degree, std::get<plssvm::real_type>(gamma), coef0);
            } else if (std::get<plssvm::gamma_coefficient_type>(gamma) == plssvm::gamma_coefficient_type::automatic) {
                return plssvm::kernel_function<plssvm::kernel_function_type::polynomial>(x, y, degree, plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()), coef0);
            } else {
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the polynomial kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("degree") = default_params.degree,
        py::arg("gamma") = default_params.gamma,
        py::arg("coef0") = default_params.coef0);
    m.def(
        "rbf_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const plssvm::gamma_type gamma) {
            if (std::holds_alternative<plssvm::real_type>(gamma)) {
                return plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x, y, std::get<plssvm::real_type>(gamma));
            } else if (std::get<plssvm::gamma_coefficient_type>(gamma) == plssvm::gamma_coefficient_type::automatic) {
                return plssvm::kernel_function<plssvm::kernel_function_type::rbf>(x, y, plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
            } else {
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the radial basis function kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("gamma") = default_params.gamma);
    m.def(
        "sigmoid_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const plssvm::gamma_type gamma, const plssvm::real_type coef0) {
            if (std::holds_alternative<plssvm::real_type>(gamma)) {
                return plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(x, y, std::get<plssvm::real_type>(gamma), coef0);
            } else if (std::get<plssvm::gamma_coefficient_type>(gamma) == plssvm::gamma_coefficient_type::automatic) {
                return plssvm::kernel_function<plssvm::kernel_function_type::sigmoid>(x, y, plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()), coef0);
            } else {
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the sigmoid kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("gamma") = default_params.gamma,
        py::arg("coef0") = default_params.coef0);
    m.def(
        "laplacian_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const plssvm::gamma_type gamma) {
            if (std::holds_alternative<plssvm::real_type>(gamma)) {
                return plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(x, y, std::get<plssvm::real_type>(gamma));
            } else if (std::get<plssvm::gamma_coefficient_type>(gamma) == plssvm::gamma_coefficient_type::automatic) {
                return plssvm::kernel_function<plssvm::kernel_function_type::laplacian>(x, y, plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
            } else {
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the laplacian kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("gamma") = default_params.gamma);
    m.def(
        "chi_squared_kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, const plssvm::gamma_type gamma) {
            if (std::holds_alternative<plssvm::real_type>(gamma)) {
                return plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(x, y, std::get<plssvm::real_type>(gamma));
            } else if (std::get<plssvm::gamma_coefficient_type>(gamma) == plssvm::gamma_coefficient_type::automatic) {
                return plssvm::kernel_function<plssvm::kernel_function_type::chi_squared>(x, y, plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size()));
            } else {
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the chi-squared kernel function to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("gamma") = default_params.gamma);

    m.def(
        "kernel_function", [](const std::vector<plssvm::real_type> &x, const std::vector<plssvm::real_type> &y, plssvm::parameter params) {
            if (params.kernel_type == plssvm::kernel_function_type::linear) {
                // gamma doesn't matter in the linear kernel function -> simply call the kernel
                return plssvm::kernel_function(x, y, params);
            } else if (std::holds_alternative<plssvm::real_type>(params.gamma)) {
                // the gamma value matters, but already is a real_type -> simply call the kernel
                return plssvm::kernel_function(x, y, params);
            } else if (std::get<plssvm::gamma_coefficient_type>(params.gamma) == plssvm::gamma_coefficient_type::automatic) {
                // the gamma value matters and is automatic -> convert it to a real_type
                params.gamma = plssvm::real_type{ 1.0 } / static_cast<plssvm::real_type>(x.size());
                return plssvm::kernel_function(x, y, params);
            } else {
                // the gamma value matters and is scale -> not supported
                throw py::value_error{ "Can't use the 'scale' gamma option since the required variance can't be calculated!" };
            }
        },
        "apply the kernel function defined in the parameter object to two vectors",
        py::arg("x"),
        py::arg("y"),
        py::kw_only(),
        py::arg("params") = default_params);

    m.def("get_gamma_type", []() {
        return plssvm::gamma_type{ plssvm::gamma_coefficient_type::automatic };
    });
}
