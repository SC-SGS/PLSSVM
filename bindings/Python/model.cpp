#include "plssvm/model.hpp"
#include "plssvm/detail/type_list.hpp"  // plssvm::detail::real_type_label_type_combination_list

#include "utility.hpp"  // assemble_unique_class_name

#include "fmt/core.h"           // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::return_value_policy
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::tuple_element_t, std::tuple_size_v
#include <utility>  // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

template <typename real_type, typename label_type>
void instantiate_model_bindings(py::module_ &m, plssvm::detail::real_type_label_type_combination<real_type, label_type>) {
    using model_type = plssvm::model<real_type, label_type>;

    const std::string class_name = assemble_unique_class_name<real_type, label_type>("Model");

    // TODO: change def to def_property_readonly based on sklearn.svm.SVC?

    py::class_<model_type>(m, class_name.c_str())
        .def(py::init<const std::string &>(), "load a previously learned model from a file")
        .def("save", &model_type::save, "save the current model to a file")
        .def("num_support_vectors", &model_type::num_support_vectors, "the number of support vectors (note: all training points become support vectors for LSSVMs)")
        .def("num_features", &model_type::num_features, "the number of features of the support vectors")
        .def("get_params", &model_type::get_params, py::return_value_policy::reference_internal, "the SVM parameter used to learn this model")
        .def("support_vectors", &model_type::support_vectors, py::return_value_policy::reference_internal, "the support vectors (note: all training points become support vectors for LSSVMs)")
        .def("weights", &model_type::weights, py::return_value_policy::reference_internal, "the weights learned for each support vector")
        .def("rho", &model_type::rho, "the bias value after learning")
        .def("__repr__", [class_name](const model_type &model) {
            return fmt::format("<plssvm.{} with {{ #sv: {}, #features: {}, rho: {} }}>",
                               class_name,
                               model.num_support_vectors(),
                               model.num_features(),
                               model.rho());
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
    instantiate_model_bindings<plssvm::detail::real_type_label_type_combination_list>(m);

    // create alias
    m.attr("Model") = m.attr(assemble_unique_class_name<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("Model").c_str());
}