/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model/regression_model.hpp"  // plssvm::regression_model

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::label_type_list
#include "plssvm/matrix.hpp"            // plssvm::aos_matrix
#include "plssvm/model/model.hpp"       // plssvm::model

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{assemble_unique_class_name, instantiate_bindings}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <string>  // std::string

namespace py = pybind11;

/**
 * @brief Functor to instantiate all regression model bindings.
 * @tparam label_type the label type for the regression model
 */
template <typename label_type>
struct regression_model_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] m the Python module
     */
    void operator()(py::module_ &m, label_type) {
        using model_type = plssvm::regression_model<label_type>;

        const std::string class_name = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("RegressionModel");

        py::class_<model_type, plssvm::model<label_type>>(m, class_name.c_str())
            .def(py::init<const std::string &>(), "load a previously learned regression model from a file")
            .def("__repr__", [class_name](const model_type &self) { return fmt::format("<plssvm.{} with {{ #sv: {}, #features: {}, rho: {} }}>",
                                                                                       class_name,
                                                                                       self.num_support_vectors(),
                                                                                       self.num_features(),
                                                                                       fmt::format("[{}]", fmt::join(self.rho(), ","))); });
    }
};

void init_regression_model(py::module_ &m) {
    // bind all regression model classes
    plssvm::bindings::python::util::instantiate_bindings<regression_model_bindings, plssvm::detail::supported_label_types_regression>(m);

    // create alias
    m.attr("RegressionModel") = m.attr(plssvm::bindings::python::util::assemble_unique_class_name<double>("RegressionModel").c_str());
}
