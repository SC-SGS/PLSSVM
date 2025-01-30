/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/min_max_scaler.hpp"  // plssvm::min_max_scaler

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "bindings/Python/conversion_to_python.hpp"  // plssvm::bindings::python::util::vector_to_pyarray

#include "fmt/format.h"         // fmt::format
#include "pybind11/numpy.h"     // py::array
#include "pybind11/pybind11.h"  // PYBIND11_NUMPY_DTYPE, py::module_, py::class_, py::init, py::arg
#include "pybind11/pytypes.h"   // py::type
#include "pybind11/stl.h"       // support for STL types

#include <array>     // std::array
#include <cstddef>   // std::size_t
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string

namespace py = pybind11;

void init_min_max_scaler(py::module_ &m) {
    PYBIND11_NUMPY_DTYPE(plssvm::min_max_scaler::factors, feature, lower, upper);

    // bind the plssvm::min_max_scaler::factors struct
    py::class_<plssvm::min_max_scaler::factors>(m, "MinMaxScalerFactors")
        .def(py::init<std::size_t, plssvm::real_type, plssvm::real_type>(), "create a new scaling factor", py::arg("feature"), py::arg("lower"), py::arg("upper"))
        .def_readonly("feature", &plssvm::min_max_scaler::factors::feature, "the feature index for which the factors are valid")
        .def_readonly("lower", &plssvm::min_max_scaler::factors::lower, "the lower scaling factor")
        .def_readonly("upper", &plssvm::min_max_scaler::factors::upper, "the upper scaling factor")
        .def("__repr__", [](const plssvm::min_max_scaler::factors &self) {
            return fmt::format("<plssvm.MinMaxScalerFactors with {{ feature: {}, lower: {}, upper: {} }}>",
                               self.feature,
                               self.lower,
                               self.upper);
        });

    // bind the plssvm::min_max_scaler class
    py::class_<plssvm::min_max_scaler>(m, "MinMaxScaler")
        .def(py::init<plssvm::real_type, plssvm::real_type>(), "create new scaling factors for the range [lower, upper]", py::arg("lower"), py::arg("upper"))
        .def(py::init([](const std::array<plssvm::real_type, 2> interval) {
                 return plssvm::min_max_scaler{ interval[0], interval[1] };
             }),
             "create new scaling factors for the range [lower, upper]")
        .def(py::init<const std::string &>(), "read the scaling factors from the file")
        .def(py::init([](const py::tuple interval) {
                 if (interval.size() != 2) {
                     throw py::value_error{ fmt::format("MinMaxScaler can only be created from two interval values (lower, upper), but {} were provided!", interval.size()) };
                 }
                 return plssvm::min_max_scaler{ interval[0].cast<plssvm::real_type>(), interval[1].cast<plssvm::real_type>() };
             }), "create new scaling factors for the range [lower, upper]")
        .def("save", &plssvm::min_max_scaler::save, "save the scaling factors to a file")
        .def("scaling_interval", &plssvm::min_max_scaler::scaling_interval, "the interval to which the data points are scaled")
        .def(
            "scaling_factors", [](const plssvm::min_max_scaler &self) -> std::optional<py::array> {
                const auto scaling_factors = self.scaling_factors();
                if (scaling_factors.has_value()) {
                    return plssvm::bindings::python::util::vector_to_pyarray(scaling_factors.value());
                } else {
                    return std::nullopt;
                }
            },
            "the scaling factors for each feature")
        .def("__repr__", [](const plssvm::min_max_scaler &self) {
            std::string optional_repr{};
            const auto scaling_factors = self.scaling_factors();
            if (scaling_factors.has_value()) {
                optional_repr += fmt::format(", #factors: {}", scaling_factors->size());
            }
            return fmt::format("<plssvm.MinMaxScaler with {{ lower: {}, upper: {}{} }}>",
                               self.scaling_interval().first,
                               self.scaling_interval().second,
                               optional_repr);
        });
}
