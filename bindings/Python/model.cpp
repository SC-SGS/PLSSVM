#include "plssvm/model.hpp"
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "utility.hpp"  // numpy_name_mapping, types_to_class_name_extension

#include "fmt/core.h"           // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::return_value_policy
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string

namespace py = pybind11;

template <typename real_type, typename label_type>
void instantiate_model_bindings_real_type_label_type(py::module_ &m) {
    using model_type = plssvm::model<real_type, label_type>;

    const std::string class_name = types_to_class_name_extension<real_type, label_type>("Model");

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

template <typename real_type>
void instantiate_model_bindings_real_type(py::module_ &m) {
    // instantiate for all supported label types (except char -> no direct numpy mapping)
    instantiate_model_bindings_real_type_label_type<real_type, bool>(m);
    instantiate_model_bindings_real_type_label_type<real_type, signed char>(m);
    instantiate_model_bindings_real_type_label_type<real_type, unsigned char>(m);
    instantiate_model_bindings_real_type_label_type<real_type, short>(m);
    instantiate_model_bindings_real_type_label_type<real_type, unsigned short>(m);
    instantiate_model_bindings_real_type_label_type<real_type, int>(m);
    instantiate_model_bindings_real_type_label_type<real_type, unsigned int>(m);
    instantiate_model_bindings_real_type_label_type<real_type, long>(m);
    instantiate_model_bindings_real_type_label_type<real_type, unsigned long>(m);
    instantiate_model_bindings_real_type_label_type<real_type, long long>(m);
    instantiate_model_bindings_real_type_label_type<real_type, unsigned long long>(m);
    instantiate_model_bindings_real_type_label_type<real_type, float>(m);
    instantiate_model_bindings_real_type_label_type<real_type, double>(m);
    instantiate_model_bindings_real_type_label_type<real_type, std::string>(m);
}

void init_model(py::module_ &m) {
    // bind all model classes

    // instantiate for all supported real types
    instantiate_model_bindings_real_type<float>(m);
    instantiate_model_bindings_real_type<double>(m);

    // create alias
    m.attr("Model") = m.attr(types_to_class_name_extension<PLSSVM_PYTHON_BINDINGS_PREFERRED_REAL_TYPE, PLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE>("Model").c_str());
}