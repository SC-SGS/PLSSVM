/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model/classification_model.hpp"  // plssvm::classification_model

#include "plssvm/constants.hpp"           // plssvm::real_type
#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t
#include "plssvm/matrix.hpp"              // plssvm::aos_matrix

#include "bindings/Python/conversion_to_python.hpp"   // plssvm::bindings::python::util::{matrix_to_pyarray, vector_to_pyarray}
#include "bindings/Python/model/variant_wrapper.hpp"  // plssvm::bindings::python::util::classification_model_wrapper
#include "bindings/Python/utility.hpp"                // plssvm::bindings::python::util::{python_type_name_mapping, create_instance}

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::pos_only, py::array, py::list
#include "pybind11/pytypes.h"   // py::type
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <memory>    // std::make_unique
#include <optional>  // std::optional, std::make_optional, std::nullopt
#include <string>    // std::string
#include <variant>   // std::visit

namespace py = pybind11;

void init_classification_model(py::module_ &m) {
    using plssvm::bindings::python::util::classification_model_wrapper;

    py::class_<classification_model_wrapper>(m, "ClassificationModel")
        .def(py::init([](const std::string &filename, const std::optional<py::type> type) {
                 if (type.has_value()) {
                     return std::make_unique<classification_model_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::classification_model, typename classification_model_wrapper::possible_model_types>(type.value(), filename));
                 } else {
                     return std::make_unique<classification_model_wrapper>(plssvm::classification_model<std::string>{ filename });
                 }
             }),
             "load a previously learned classification model from a file",
             py::arg("filename"),
             py::pos_only(),
             py::arg("type") = std::nullopt)
        .def("save", [](const classification_model_wrapper &self, const std::string &filename) { return std::visit([&filename](auto &&model) { model.save(filename); }, self.model); }, "save the current model to a file")
        .def("num_support_vectors", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return model.num_support_vectors(); }, self.model); }, "the number of support vectors (note: all training points become support vectors for LSSVMs)")
        .def("num_features", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return model.num_features(); }, self.model); }, "the number of features of the support vectors")
        .def("get_params", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return model.get_params(); }, self.model); }, "the SVM parameter used to learn this model")
        .def("support_vectors", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return plssvm::bindings::python::util::matrix_to_pyarray(model.support_vectors()); }, self.model); }, "the support vectors (note: all training points become support vectors for LSSVMs)")
        // clang-format off
        .def("labels", [](const classification_model_wrapper &self) {
            return std::visit([](auto &&model) -> std::optional<py::array> {
                if (model.labels().has_value()) {
                    return std::make_optional(plssvm::bindings::python::util::vector_to_pyarray(model.labels()->get()));
                } else {
                    return std::nullopt;
                }
            }, self.model); }, "the labels")
        .def("weights", [](const classification_model_wrapper &self) {
            return std::visit([](auto &&model) {
                py::list ret{};
                for (const plssvm::aos_matrix<plssvm::real_type> &matr : model.weights()) {
                    ret.append(plssvm::bindings::python::util::matrix_to_pyarray(matr));
                }
                return ret;
            }, self.model); }, "the weights learned for each support vector and class")
        // clang-format on
        .def("rho", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(model.rho()); }, self.model); }, "the bias value after learning for each class")
        .def("num_classes", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return model.num_classes(); }, self.model); }, "the number of classes")
        .def("classes", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(model.classes()); }, self.model); }, "the classes")
        .def("get_classification_type", [](const classification_model_wrapper &self) { return std::visit([](auto &&model) { return model.get_classification_type(); }, self.model); }, "the classification type used to create this model")
        // clang-format off
        .def("__repr__", [](const classification_model_wrapper &self) {
            return std::visit([](auto &&model) {
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                return fmt::format("<plssvm.ClassificationModel with {{ label_type: {}, #sv: {}, #features: {}, rho: [{}], classification_type: {} }}>",
                            plssvm::bindings::python::util::python_type_name_mapping<label_type>(),
                            model.num_support_vectors(),
                            model.num_features(),
                            fmt::join(model.rho(), ","),
                            model.get_classification_type());
            }, self.model); });
    // clang-format on
}
