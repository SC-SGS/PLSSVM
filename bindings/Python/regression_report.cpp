/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/regression_report.hpp"  // plssvm::regression_report

#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t

#include "bindings/Python/data_set/variant_wrapper.hpp"                 // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/type_caster/label_vector_wrapper_caster.hpp"  // a custom Pybind11 type caster for a plssvm::bindings::python::util::label_vector_wrapper

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::init, py::arg, py::pos_only, py::value_error
#include "pybind11/pytypes.h"   // py::object
#include "pybind11/stl.h"       // support for STL types

#include <string>   // std::string
#include <variant>  // std::visit, std::get

namespace py = pybind11;

void init_regression_report(py::module_ &m) {
    // bind regression_report class
    m.def("regression_report", [](plssvm::bindings::python::util::label_vector_wrapper<typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_vector_types> y_true, plssvm::bindings::python::util::label_vector_wrapper<typename plssvm::bindings::python::util::regression_data_set_wrapper::possible_vector_types> y_pred, const bool force_finite, const bool output_dict) -> py::object {
        using plssvm::bindings::python::util::regression_data_set_wrapper;

        // check that the data types are equal
        if (!y_true.dtype.equal(y_pred.dtype)) {
            throw py::value_error{ fmt::format(R"(The type of the correct labels "{}" differs from the type of the predicted labels "{}"!)", y_true.dtype.attr("name").cast<std::string>(), y_pred.dtype.attr("name").cast<std::string>()) };
        }

         return std::visit([&](auto &&correct_label) -> py::object {
             using vector_type = plssvm::detail::remove_cvref_t<decltype(correct_label)>;
             const plssvm::regression_report report{ correct_label, std::get<vector_type>(y_pred.labels), plssvm::regression_report::force_finite = force_finite };

             if (output_dict) {
                 // get the metrics
                 const plssvm::regression_report::metric metrics = report.loss();

                 // fill the Python dictionary
                 py::dict dict{};
                 dict["explained_variance_score"] = metrics.explained_variance_score;
                 dict["mean_absolute_error"] = metrics.mean_absolute_error;
                 dict["mean_squared_error"] = metrics.mean_squared_error;
                 dict["r2_score"] = metrics.r2_score;
                 dict["squared_correlation_coefficient"] = metrics.squared_correlation_coefficient;
                 return dict;
             } else {
                 return py::str(fmt::format("{}", report));
             }
         },
                           y_true.labels); }, "create a new regression report by calculating all metrics between the correct and predicted labels", py::arg("y_true"), py::arg("y_pred"), py::pos_only(), py::arg("force_finite") = true, py::arg("output_dict") = false);
}
