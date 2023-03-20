#include "plssvm/model.hpp"

#include "fmt/core.h"           // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::return_value_policy
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string

namespace py = pybind11;

void init_model(py::module_ &m) {
    // bind model class
    using real_type = double;
    using label_type = std::string;
    using model_type = plssvm::model<real_type, label_type>;

    // TODO: change def to def_property_readonly based on sklearn.svm.SVC?

    py::class_<model_type>(m, "Model")
        .def(py::init<const std::string &>(), "load a previously learned model from a file")
        .def("save", &model_type::save, "save the current model to a file")
        .def("num_support_vectors", &model_type::num_support_vectors, "the number of support vectors (note: all training points become support vectors for LSSVMs)")
        .def("num_features", &model_type::num_features, "the number of features of the support vectors")
        .def("get_params", &model_type::get_params, py::return_value_policy::reference_internal, "the SVM parameter used to learn this model")
        .def("support_vectors", &model_type::support_vectors, py::return_value_policy::reference_internal, "the support vectors (note: all training points become support vectors for LSSVMs)")
        .def("weights", &model_type::weights, py::return_value_policy::reference_internal, "the weights learned for each support vector")
        .def("rho", &model_type::rho, "the bias value after learning")
        .def("__repr__", [](const model_type &model) {
            return fmt::format("<plssvm.Model with {{ #sv: {}, #features: {}, rho: {} }}>",
                               model.num_support_vectors(), model.num_features(), model.rho());
        });
}