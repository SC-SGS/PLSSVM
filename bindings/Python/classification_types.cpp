/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/classification_types.hpp"  // plssvm::classification_type, plssvm::classification_type_to_full_string, plssvm::calculate_number_of_classifiers

#include "bindings/Python/bindings_fwd.hpp"  // forward declare all helper functions to create the Python bindings
#include "bindings/Python/utility.hpp"       // plssvm::bindings::python::util::register_implicit_str_enum_conversion

#include "pybind11/cast.h"      // py::arg
#include "pybind11/pybind11.h"  // py::module_, py::arg, py::enum_

namespace py = pybind11;

void init_classification_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::classification_type> py_enum(m, "ClassificationType", "enum.Enum", "Enum class for all implemented multiclass classification strategies.");
    py_enum
        .value("OAA", plssvm::classification_type::oaa, "use the one vs. all classification strategy (default)")
        .value("OAO", plssvm::classification_type::oao, "use the one vs. one classification strategy");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::classification_type>(py_enum);

    // bind free functions
    m.def("classification_type_to_full_string", &plssvm::classification_type_to_full_string, "convert the classification type to its full string representation", py::arg("classification"));
    m.def("calculate_number_of_classifiers", &plssvm::calculate_number_of_classifiers, "given the classification strategy and number of classes , calculates the number of necessary classifiers", py::arg("classification"), py::arg("num_classes"));
}
