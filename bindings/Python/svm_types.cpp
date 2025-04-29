/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm_types.hpp"  // plssvm::svm_type, plssvm::list_available_svm_types, plssvm::svm_type_from_model_file

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::register_implicit_str_enum_conversion

#include "pybind11/pybind11.h"  // py::module_, py::enum_

namespace py = pybind11;

void init_svm_types(py::module_ &m) {
    // bind enum class
    py::enum_<plssvm::svm_type> py_enum(m, "SVMType", "Enum class for all implemented SVM types in PLSSVM.");
    py_enum
        .value("CSVC", plssvm::svm_type::csvc, "use a C-SVC for classification")
        .value("CSVR", plssvm::svm_type::csvr, "use a C-SVR for classification");

    // enable implicit conversion from string to enum
    plssvm::bindings::python::util::register_implicit_str_enum_conversion<plssvm::svm_type>(py_enum);

    // bind free functions
    m.def("list_available_svm_types", &plssvm::list_available_svm_types, "list the available SVM types");
    m.def("svm_type_to_task_name", &plssvm::svm_type_to_task_name, "get the task name (e.g., \"classification\" or \"regression\") based on the provided SVMType", py::arg("svm_type"));
    m.def("svm_type_from_model_file", &plssvm::svm_type_from_model_file, "determine the SVMType based on the provided LIBSVM model file", py::arg("filename"));
}
