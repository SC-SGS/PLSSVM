/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model/classification_model.hpp"  // plssvm::classification_model

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::label_type_list
#include "plssvm/matrix.hpp"            // plssvm::aos_matrix
#include "plssvm/model/model.hpp"       // plssvm::model

#include "bindings/Python/utility.hpp"  // assemble_unique_class_name, vector_to_pyarray, instantiate_bindings

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace py = pybind11;

/**
 * @brief Functor to instantiate all classification model bindings.
 * @tparam label_type the label type for the classification model
 */
template <typename label_type>
struct classification_model_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] m the Python module
     */
    void operator()(py::module_ &m, label_type) {
        using model_type = plssvm::classification_model<label_type>;

        const std::string class_name = assemble_unique_class_name<label_type>("ClassificationModel");

        py::class_<model_type, plssvm::model<label_type>>(m, class_name.c_str())
            .def(py::init<const std::string &>(), "load a previously learned classification model from a file")
            .def("num_classes", &model_type::num_classes, "the number of classes")
            .def("classes", [](const model_type &self) {
               if constexpr (std::is_same_v<label_type, std::string>) {
                   return self.classes();
               } else {
                   return vector_to_pyarray(self.classes());
               } }, "the classes")
            .def("get_classification_type", [](const model_type &self) { return self.get_classification_type(); }, "the classification type used to create this model")
            .def("__repr__", [class_name](const model_type &self) { return fmt::format("<plssvm.{} with {{ #sv: {}, #features: {}, rho: {}, classification_type: {} }}>",
                                                                                       class_name,
                                                                                       self.num_support_vectors(),
                                                                                       self.num_features(),
                                                                                       fmt::format("[{}]", fmt::join(self.rho(), ",")),
                                                                                       self.get_classification_type()); });
    }
};

void init_classification_model(py::module_ &m) {
    // bind all classification model classes
    instantiate_bindings<classification_model_bindings, plssvm::detail::supported_label_types_classification>(m);

    // create alias
    m.attr("ClassificationModel") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_SVC_LABEL_TYPE>("ClassificationModel").c_str());
}
