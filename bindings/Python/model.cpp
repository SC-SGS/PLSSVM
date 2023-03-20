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

    py::class_<model_type>(m, "model")
        .def(py::init<const std::string &>())
        .def("save", &model_type::save)
        .def("num_support_vectors", &model_type::num_support_vectors)
        .def("num_features", &model_type::num_features)
        .def("get_params", &model_type::get_params, py::return_value_policy::reference_internal)
        .def("support_vectors", &model_type::support_vectors, py::return_value_policy::reference_internal)
        .def("weights", &model_type::weights, py::return_value_policy::reference_internal)
        .def("rho", &model_type::rho)
        .def("__repr__", [](const model_type &model) {
            return fmt::format("<plssvm.model with {{ #sv: {}, #features: {}, rho: {} }}>",
                               model.num_support_vectors(), model.num_features(), model.rho());
        });
}