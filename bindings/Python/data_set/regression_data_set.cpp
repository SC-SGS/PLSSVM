/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set

#include "plssvm/data_set/min_max_scaler.hpp"  // plssvm::min_max_scaler
#include "plssvm/detail/type_traits.hpp"       // plssvm::detail::remove_cvref_t
#include "plssvm/file_format_types.hpp"        // plssvm::file_format_type

#include "bindings/Python/conversion_from_python.hpp"    // plssvm::bindings::python::util::{pyobject_to_matrix, pyobject_to_vector}
#include "bindings/Python/conversion_to_python.hpp"      // plssvm::bindings::python::util::{matrix_to_pyarray, vector_to_pyarray}
#include "bindings/Python/data_set/variant_wrapper.hpp"  // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/utility.hpp"                   // plssvm::bindings::python::util::{create_instance, python_type_name_mapping}

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/numpy.h"     // py::array_t, py::array
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::pos_only, py::object, py::attribute_error
#include "pybind11/pytypes.h"   // py::type
#include "pybind11/stl.h"       // support for STL types

#include <memory>    // std::make_unique
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::move
#include <variant>   // std::visit

namespace py = pybind11;

void init_regression_data_set(py::module_ &m) {
    using plssvm::bindings::python::util::regression_data_set_wrapper;

    py::class_<regression_data_set_wrapper>(m, "RegressionDataSet")
        .def(py::init([](const std::string &filename, const std::optional<py::type> type, const plssvm::file_format_type format, const std::optional<plssvm::min_max_scaler> scaler) {
                 if (type.has_value()) {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), filename, format, scaler.value()));
                     } else {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), filename, format));
                     }
                 } else {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ filename, format, scaler.value() });
                     } else {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ filename, format });
                     }
                 }
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("filename"),
             py::pos_only(),
             py::arg("type") = std::nullopt,
             py::arg("format") = plssvm::file_format_type::libsvm,
             py::arg("scaler") = std::nullopt)
        .def(py::init([](py::object data, const std::optional<py::type> type, const std::optional<plssvm::min_max_scaler> scaler) {
                 // convert the data py::object to a plssvm::aos_matrix
                 const auto &[data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);

                 if (type.has_value()) {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(data_matrix), scaler.value()));
                     } else {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(data_matrix)));
                     }
                 } else {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(data_matrix), scaler.value() });
                     } else {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(data_matrix) });
                     }
                 }
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("X"),
             py::pos_only(),
             py::arg("type") = std::nullopt,
             py::arg("scaler") = std::nullopt)
        .def(py::init([](py::object data, py::object labels, const std::optional<plssvm::min_max_scaler> scaler) {
                 // convert the data py::object to a plssvm::aos_matrix
                 auto [data_matrix, opt_feature_names] = plssvm::bindings::python::util::pyobject_to_matrix(data);
                 // convert the labels to a std::vector
                 auto [labels_vector_variant, dtype] = plssvm::bindings::python::util::pyobject_to_vector<typename regression_data_set_wrapper::possible_vector_types>(labels);

                 return std::visit([&dtype, &data_matrix, &scaler](auto &&labels_vector) {
                     using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<label_type>(std::move(data_matrix), std::move(labels_vector), scaler.value()));
                     } else {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<label_type>(std::move(data_matrix), std::move(labels_vector)));
                     }
                 },
                                   labels_vector_variant);
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("X"),
             py::arg("y"),
             py::pos_only(),
             py::arg("scaler") = std::nullopt)
        .def("save", [](const regression_data_set_wrapper &self, const std::string &filename, const plssvm::file_format_type format) { std::visit([&filename, format](auto &&data) { data.save(filename, format); }, self.data_set); }, "save the data set to a file using the provided file format type", py::arg("filename"), py::pos_only(), py::arg("format") = plssvm::file_format_type::libsvm)
        .def("data", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return plssvm::bindings::python::util::matrix_to_pyarray(data.data()); }, self.data_set); }, "the data saved as 2D vector")
        .def("has_labels", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.has_labels(); }, self.data_set); }, "check whether the data set has labels")
        // clang-format off
        .def("labels", [](const regression_data_set_wrapper &self) {
            return std::visit([](auto &&data) {
                if (!data.has_labels()) {
                    throw py::attribute_error{ "'RegressionDataSet' object has no function 'labels'. Maybe this RegressionDataSet was created without labels?" };
                } else {
                    return plssvm::bindings::python::util::vector_to_pyarray(data.labels()->get());
                }
            }, self.data_set); }, "the labels")
        // clang-format on
        .def("num_data_points", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.num_data_points(); }, self.data_set); }, "the number of data points in the data set")
        .def("num_features", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.num_features(); }, self.data_set); }, "the number of features per data point")
        .def("is_scaled", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.is_scaled(); }, self.data_set); }, "check whether the original data has been scaled to [lower, upper] bounds")
        .def("scaling_factors", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) {
            if (!data.is_scaled()) {
                throw py::attribute_error{ "'RegressionDataSet' object has no function 'scaling_factors'. Maybe this RegressionDataSet has not been scaled?" };
            } else {
                return data.scaling_factors().value();
            } }, self.data_set); }, py::return_value_policy::reference_internal, "the factors used to scale this data set")
        // clang-format off
        .def("__repr__", [](const regression_data_set_wrapper &self) {
            return std::visit([](auto &&data) {
                std::string optional_repr{};
                if (data.is_scaled()) {
                    optional_repr += fmt::format(", scaling: [{}, {}]",
                                        data.scaling_factors()->get().scaling_interval().first,
                                        data.scaling_factors()->get().scaling_interval().second);
                }
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type;
                return fmt::format("<plssvm.RegressionDataSet with {{ label_type: {}, #points: {}, #features: {}{} }}>",
                                        plssvm::bindings::python::util::python_type_name_mapping<label_type>(),
                                        data.num_data_points(),
                                        data.num_features(),
                                        optional_repr);
            }, self.data_set); });
    // clang-format on
}
