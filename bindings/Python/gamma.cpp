/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/gamma.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/matrix.hpp"     // plssvm::aos_matrix

#include "bindings/Python/type_caster/matrix_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::matrix

#include "pybind11/pybind11.h"  // py::module_, py::enum_
#include "pybind11/stl.h"       // support for STL types: std::variant

namespace py = pybind11;

void init_gamma(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::gamma_coefficient_type>(m, "GammaCoefficientType", "Enum class for all possible gamma coefficient types (can also be a number).")
        .value("AUTOMATIC", plssvm::gamma_coefficient_type::automatic, "use a dynamic gamma value of 1 / num_features for the kernel functions")
        .value("SCALE", plssvm::gamma_coefficient_type::scale, "use a dynamic gamma value of 1 / (num_features * data.var()) for the kernel functions");

    // bind free functions
    m.def("get_gamma_string", &plssvm::get_gamma_string, "get the gamma string based on the currently active variant member", py::arg("gamma"));
    m.def(
        "calculate_gamma_value", [](const plssvm::gamma_type &gamma, const plssvm::aos_matrix<plssvm::real_type> &data) {
            return plssvm::calculate_gamma_value(gamma, data);
        },
        "get the real_type value of the provided gamma type",
        py::arg("gamma"),
        py::arg("matrix"));
}
