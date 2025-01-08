/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model/model.hpp"  // plssvm::model

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::label_type_list
#include "plssvm/matrix.hpp"            // plssvm::aos_matrix

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{assemble_unique_class_name, vector_to_pyarray, matrix_to_pyarray, instantiate_module_bindings}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::return_value_policy, py::list
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <optional>     // std::make_optional, std::nullopt
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move

namespace py = pybind11;

/**
 * @brief Functor to instantiate all base model bindings.
 * @tparam label_type the label type for the base model
 */
template <typename label_type>
struct model_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] m the Python module
     */
    void operator()(py::module_ &m, label_type) {
        using model_type = plssvm::model<label_type>;

        const std::string class_name = plssvm::bindings::python::util::assemble_unique_class_name<label_type>("__pure_virtual_base_Model");

        py::class_<model_type> py_model(m, class_name.c_str());
        py_model.def("save", &model_type::save, "save the current model to a file")
            .def("num_support_vectors", &model_type::num_support_vectors, "the number of support vectors (note: all training points become support vectors for LSSVMs)")
            .def("num_features", &model_type::num_features, "the number of features of the support vectors")
            .def("get_params", &model_type::get_params, py::return_value_policy::reference_internal, "the SVM parameter used to learn this model")
            .def("support_vectors", [](const model_type &self) { return plssvm::bindings::python::util::matrix_to_pyarray(self.support_vectors()); }, "the support vectors (note: all training points become support vectors for LSSVMs)")
            .def("weights", []([[maybe_unused]] const model_type &self) {
                py::list ret{};
                for (const plssvm::aos_matrix<plssvm::real_type> &matr : self.weights()) {
                    ret.append(plssvm::bindings::python::util::matrix_to_pyarray(matr));
                }
                return ret; }, "the weights learned for each support vector and class")
            .def("rho", [](const model_type &self) { return plssvm::bindings::python::util::vector_to_pyarray(self.rho()); }, "the bias value after learning for each class");
        if constexpr (std::is_same_v<label_type, std::string>) {
            py_model.def("labels", [](const model_type &self) -> std::optional<py::list> {
                if (self.labels().has_value()) {
                    py::list ret{};
                    for (std::string str : self.labels()->get()) {
                        ret.append(std::move(str));
                    }
                    return std::make_optional(ret);
                } else {
                    return std::nullopt;
                } }, "the labels");
        } else {
            py_model.def("labels", [](const model_type &self) -> std::optional<py::array_t<label_type, 1>> {
                if (self.labels().has_value()) {
                    return std::make_optional(plssvm::bindings::python::util::vector_to_pyarray(self.labels()->get()));
                } else {
                    return std::nullopt;
                } }, "the labels");
        }
    }
};

void init_model(py::module_ &pure_virtual) {
    // bind all pure-virtual base model classes
    // NOTE: supported_label_types_classification also contains all types in supported_label_types_regression
    plssvm::bindings::python::util::instantiate_module_bindings<model_bindings, plssvm::detail::supported_label_types_classification>(pure_virtual);
}
