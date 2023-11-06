/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model.hpp"

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::label_type_list

#include "utility.hpp"                  // assemble_unique_class_name, vector_to_pyarray, matrix_to_pyarray

#include "fmt/core.h"                   // fmt::format
#include "pybind11/pybind11.h"          // py::module_, py::class_, py::return_value_policy
#include "pybind11/stl.h"               // support for STL types: std::vector

#include <cstddef>                      // std::size_t
#include <string>                       // std::string
#include <tuple>                        // std::tuple_element_t, std::tuple_size_v
#include <type_traits>                  // std::is_same_v
#include <utility>                      // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

template <typename label_type>
void instantiate_model_bindings(py::module_ &m, label_type) {
    using model_type = plssvm::model<label_type>;

    const std::string class_name = assemble_unique_class_name<label_type>("Model");

    py::class_<model_type>(m, class_name.c_str())
        .def(py::init<const std::string &>(), "load a previously learned model from a file")
        .def("save", &model_type::save, "save the current model to a file")
        .def("num_support_vectors", &model_type::num_support_vectors, "the number of support vectors (note: all training points become support vectors for LSSVMs)")
        .def("num_features", &model_type::num_features, "the number of features of the support vectors")
        .def("get_params", &model_type::get_params, py::return_value_policy::reference_internal, "the SVM parameter used to learn this model")
        .def(
            "support_vectors", [](const model_type &self) {
                return matrix_to_pyarray(self.support_vectors());
            },
            "the support vectors (note: all training points become support vectors for LSSVMs)")
        .def(
            "labels", [](const model_type &self) {
                if constexpr (std::is_same_v<label_type, std::string>) {
                    return self.labels();
                } else {
                    return vector_to_pyarray(self.labels());
                }
            },
            "the labels")
        .def("num_classes", &model_type::num_classes, "the number of classes")
        .def(
            "classes", [](const model_type &self) {
                if constexpr (std::is_same_v<label_type, std::string>) {
                    return self.classes();
                } else {
                    return vector_to_pyarray(self.classes());
                }
            },
            "the classes")
        .def(
            "weights", []([[maybe_unused]] const model_type &self) {
                py::list ret{};
                for (const plssvm::aos_matrix<plssvm::real_type> &matr : self.weights()) {
                    ret.append(matrix_to_pyarray(matr));
                }
                return ret;
            },
            "the weights learned for each support vector and class")
        .def("rho", [](const model_type &self) {
                return vector_to_pyarray(self.rho());
            }, "the bias value after learning for each class")
        .def("get_classification_type", [](const model_type &self) {
            return self.get_classification_type();
        }, "the classification type used to create this model")
        .def("__repr__", [class_name](const model_type &self) {
            return fmt::format("<plssvm.{} with {{ #sv: {}, #features: {}, rho: {}, classification_type: {} }}>",
                               class_name,
                               self.num_support_vectors(),
                               self.num_features(),
                               fmt::format("[{}]", fmt::join(self.rho(), ",")),
                               self.get_classification_type());
        });
}

template <typename T, std::size_t... Idx>
void instantiate_model_bindings(py::module_ &m, std::integer_sequence<std::size_t, Idx...>) {
    (instantiate_model_bindings(m, std::tuple_element_t<Idx, T>{}), ...);
}

template <typename T>
void instantiate_model_bindings(py::module_ &m) {
    instantiate_model_bindings<T>(m, std::make_integer_sequence<std::size_t, std::tuple_size_v<T>>{});
}

void init_model(py::module_ &m) {
    // bind all model classes
    instantiate_model_bindings<plssvm::detail::supported_label_types>(m);

    // create alias
    m.attr("Model") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("Model").c_str());
}