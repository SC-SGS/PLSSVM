/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/regression_report.hpp"  // plssvm::regression_report

#include "plssvm/detail/type_list.hpp"  // plssvm::detail::supported_label_types_regression

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{check_kwargs_for_correctness, instantiate_class_bindings}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kwargs
#include "pybind11/stl.h"       // support for STL types

#include <vector>  // std::vector

namespace py = pybind11;

/**
 * @brief Functor to instantiate all regression report bindings.
 * @tparam label_type the label type for the regression report
 */
template <typename label_type>
struct regression_report_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] rp the Python regression report class
     */
    void operator()(py::class_<plssvm::regression_report> &rp, label_type) {
        rp.def(py::init<>([](const std::vector<label_type> &correct_label, const std::vector<label_type> &predicted_label, const py::kwargs &args) {
                   // check keyword arguments
                   plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "force_finite" });

                   if (args.contains("force_finite")) {
                       return plssvm::regression_report{ correct_label, predicted_label, plssvm::regression_report::force_finite = args["force_finite"].cast<bool>() };
                   } else {
                       return plssvm::regression_report{ correct_label, predicted_label };
                   }
               }),
               "create a new regression report by calculating all metrics between the correct and predicted labels");
    }
};

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
    py::class_<plssvm::regression_report> py_regression_report(m, "RegressionReport");
    py_regression_report
        .def("loss", &plssvm::regression_report::loss, "return the calculated regression metrics between the correct and predicted labels")
        .def("__repr__", [](const plssvm::regression_report &self) { return fmt::format("{}", self); });

    // instantiate all possible templated constructors
    plssvm::bindings::python::util::instantiate_class_bindings<regression_report_bindings, plssvm::detail::supported_label_types_regression>(py_regression_report);

    // make alias to be in line with sklearn's classification_report
    m.attr("regression_report") = m.attr("RegressionReport");
}
