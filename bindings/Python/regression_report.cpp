/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/regression_report.hpp"  // plssvm::regression_report

#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t

#include "bindings/Python/conversion_from_python.hpp"    // plssvm::bindings::python::util::pyobject_to_vector
#include "bindings/Python/data_set/variant_wrapper.hpp"  // plssvm::bindings::python::util::regression_data_set_wrapper

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::pos_only, py::value_error
#include "pybind11/pytypes.h"   // py::object
#include "pybind11/stl.h"       // support for STL types

#include <string>   // std::string
#include <variant>  // std::visit, std::get
#include <vector>   // std::vector

namespace py = pybind11;

void init_regression_report(py::module_ &m) {
    // bind regression report class
    py::class_<plssvm::regression_report::metric>(m, "RegressionReportMetric")
        .def(py::init<>())
        .def_property_readonly("explained_variance_score", [](const plssvm::regression_report::metric &self) { return self.explained_variance_score; })
        .def_property_readonly("mean_absolute_error", [](const plssvm::regression_report::metric &self) { return self.mean_absolute_error; })
        .def_property_readonly("mean_squared_error", [](const plssvm::regression_report::metric &self) { return self.mean_squared_error; })
        .def_property_readonly("r2_score", [](const plssvm::regression_report::metric &self) { return self.r2_score; })
        .def_property_readonly("squared_correlation_coefficient", [](const plssvm::regression_report::metric &self) { return self.squared_correlation_coefficient; })
        .def("__repr__", [](const plssvm::regression_report::metric &self) { return fmt::format("{}", self); });

    // bind regression_report class
    py::class_<plssvm::regression_report>(m, "RegressionReport")
        .def(py::init([](py::object y_true, py::object y_pred, const bool force_finite) {
                 using plssvm::bindings::python::util::regression_data_set_wrapper;

                 // convert the correct labels to a std::vector
                 const auto &[correct_label_variant, dtype_correct_label] = plssvm::bindings::python::util::pyobject_to_vector<typename regression_data_set_wrapper::possible_vector_types>(y_true);
                 // convert the predicted labels to a std::vector
                 const auto &[predicted_label_variant, dtype_predicted_label] = plssvm::bindings::python::util::pyobject_to_vector<typename regression_data_set_wrapper::possible_vector_types>(y_pred);

                 // check that the data types are equal
                 if (!dtype_correct_label.equal(dtype_predicted_label)) {
                     throw py::value_error{ fmt::format(R"(The type of the correct labels "{}" differs from the type of the predicted labels "{}"!)", dtype_correct_label.attr("name").cast<std::string>(), dtype_predicted_label.attr("name").cast<std::string>()) };
                 }

                 return std::visit([&predicted_label_variant, force_finite](auto &&correct_label) {
                     using vector_type = plssvm::detail::remove_cvref_t<decltype(correct_label)>;
                     return plssvm::regression_report{ correct_label, std::get<vector_type>(predicted_label_variant), plssvm::regression_report::force_finite = force_finite };
                 },
                                   correct_label_variant);
             }),
             "create a new regression report by calculating all metrics between the correct and predicted labels",
             py::arg("y_true"),
             py::arg("y_pred"),
             py::pos_only(),
             py::arg("force_finite") = true)
        .def("loss", &plssvm::regression_report::loss, "return the calculated regression metrics between the correct and predicted labels")
        .def("__repr__", [](const plssvm::regression_report &self) { return fmt::format("{}", self); });

    // make alias to be in line with sklearn's classification_report
    m.attr("regression_report") = m.attr("RegressionReport");
}
