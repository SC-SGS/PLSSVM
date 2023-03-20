#include "plssvm/data_set.hpp"

#include "utility.hpp"  // check_kwargs_for_correctness

#include "fmt/core.h"           // fmt::format
#include "fmt/format.h"         // fmt::join
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::return_value_policy, py::arg, py::kwargs, py::value_error, py::pos_only
#include "pybind11/stl.h"       // support for STL types

#include <string>   // std::string
#include <utility>  // std::move
#include <vector>   // std::vector

namespace py = pybind11;

void init_data_set(py::module_ &m) {
    // bind data_set class
    using real_type = double;
    using label_type = std::string;
    using data_set_type = plssvm::data_set<real_type, label_type>;
    using size_type = typename data_set_type::size_type;

    // TODO: change def to def_property_readonly based on sklearn.svm.SVC?

    // bind the plssvm::data_set::scaling internal "factors" struct
    py::class_<data_set_type::scaling::factors>(m, "DataSetScalingFactors")
        .def(py::init<size_type, real_type, real_type>(), "create a new scaling factor",
            py::arg("feature"), py::arg("lower"), py::arg("upper"))
        .def_readonly("feature", &data_set_type::scaling::factors::feature, "the feature index for which the factors are valid")
        .def_readonly("lower", &data_set_type::scaling::factors::lower, "the lower scaling factor")
        .def_readonly("upper", &data_set_type::scaling::factors::upper, "the upper scaling factor")
        .def("__repr__", [](const data_set_type::scaling::factors &scaling_factors) {
            return fmt::format("<plssvm.DataSetScalingFactors with {{ feature: {}, lower: {}, upper: {} }}>",
                               scaling_factors.feature,
                               scaling_factors.lower,
                               scaling_factors.upper);
        });

    // bind the plssvm::data_set internal "scaling" struct
    py::class_<data_set_type::scaling>(m, "DataSetScaling")
        .def(py::init<real_type, real_type>(), py::arg("lower"), py::arg("upper"), "create new scaling factors for the range [lower, upper]")
        .def(py::init<const std::string &>(), "read the scaling factors from the file")
        .def("save", &data_set_type::scaling::save, "save the scaling factors to a file")
        .def_readonly("scaling_interval", &data_set_type::scaling::scaling_interval, "the interval to which the data points are scaled")
        .def_readonly("scaling_factors", &data_set_type::scaling::scaling_factors, "the scaling factors for each feature")
        .def("__repr__", [](const data_set_type::scaling &scaling) {
            return fmt::format("<plssvm.DataSetScaling with {{ lower: {}, upper: {}, #factors: {} }}>",
                               scaling.scaling_interval.first,
                               scaling.scaling_interval.second,
                               scaling.scaling_factors.size());
        });

    // bind the data set class
    py::class_<data_set_type>(m, "DataSet")
        .def(py::init([](const std::string &file_name, py::kwargs args) {
            // check for valid keys
            check_kwargs_for_correctness(args, { "file_format", "scaling" });

            // call the constructor corresponding to the provided named arguments
            if (args.contains("file_format") && args.contains("scaling")) {
                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>(), args["scaling"].cast<data_set_type::scaling>() };
            } else if (args.contains("file_format")) {
                return data_set_type{ file_name, args["file_format"].cast<plssvm::file_format_type>() };
            } else if (args.contains("scaling")) {
                return data_set_type{ file_name, args["scaling"].cast<data_set_type::scaling>() };
            } else {
                return data_set_type{ file_name };
            }
        }), "create a new data set without labels given additional optional parameters")
        .def(py::init([](std::vector<std::vector<real_type>> data, py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "scaling" });

            if (args.contains("scaling")) {
                return data_set_type{ std::move(data), args["scaling"].cast<data_set_type::scaling>() };
            } else {
                return data_set_type{ std::move(data) };
            }
        }), "create a new data set with labels given additional optional parameters")
        .def(py::init([](std::vector<std::vector<real_type>> data, py::list labels, py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "scaling" });

            // TODO: investigate performance implications?
            std::vector<std::string> tmp(py::len(labels));
            #pragma omp parallel for
            for (std::vector<std::string>::size_type i = 0; i < py::len(labels); ++i) {
                tmp[i] = labels[i].cast<py::str>().cast<std::string>();
            }

            if (args.contains("scaling")) {
                return data_set_type{ std::move(data), std::move(tmp), args["scaling"].cast<data_set_type::scaling>() };
            } else {
                return data_set_type{ std::move(data), std::move(tmp) };
            }
        }))
        .def("save", &data_set_type::save, "save the data set to a file")
        .def("num_data_points", &data_set_type::num_data_points, "the number of data points in the data set")
        .def("num_features", &data_set_type::num_features, "the number of features per data point")
        .def("data", &data_set_type::data, py::return_value_policy::reference_internal, "the data saved as 2D vector")
        .def("has_labels", &data_set_type::has_labels, "check whether the data set has labels")
        .def("labels", &data_set_type::labels, py::return_value_policy::reference_internal, "the labels")
        .def("num_different_labels", &data_set_type::num_different_labels, "the number of different labels")
        .def("different_labels", &data_set_type::different_labels, "the different labels")
        .def("is_scaled", &data_set_type::is_scaled, "check whether the original data has been scaled to [lower, upper] bounds")
        .def("scaling_factors", &data_set_type::scaling_factors, py::return_value_policy::reference_internal, "the factors used to scale this data set")
        .def("__repr__", [](const data_set_type &data) {
            std::string optional_repr{};
            if (data.has_labels()) {
                optional_repr += fmt::format(", labels: [{}]", fmt::join(data.different_labels().value(), ", "));
            }
            if (data.is_scaled()) {
                optional_repr += fmt::format(", scaling: [{}, {}]",
                                             data.scaling_factors()->get().scaling_interval.first,
                                             data.scaling_factors()->get().scaling_interval.second);
            }
            return fmt::format("<plssvm.DataSet with {{ #points: {}, #features: {}{} }}>",
                               data.num_data_points(),
                               data.num_features(),
                               optional_repr);
        });
}