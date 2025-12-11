/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set

#include "plssvm/constants.hpp"                // plssvm::real_type
#include "plssvm/data_set/min_max_scaler.hpp"  // plssvm::min_max_scaler
#include "plssvm/detail/type_traits.hpp"       // plssvm::detail::remove_cvref_t
#include "plssvm/file_format_types.hpp"        // plssvm::file_format_type
#include "plssvm/matrix.hpp"                   // plssvm::soa_matrix
#include "plssvm/mpi/communicator.hpp"         // plssvm::mpi::communicator

#include "bindings/Python/bindings_fwd.hpp"                                  // forward declare all helper functions to create the Python bindings
#include "bindings/Python/data_set/variant_wrapper.hpp"                      // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/type_caster/label_vector_wrapper_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::bindings::python::util::label_vector_wrapper
#include "bindings/Python/type_caster/matrix_type_caster.hpp"                // NOLINT: a custom Pybind11 type caster for a plssvm::matrix
#include "bindings/Python/type_caster/mpi_type_caster.hpp"                   // NOLINT: a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                                       // plssvm::bindings::python::util::{create_instance, python_type_name_mapping, vector_to_pyarray}

#include "fmt/format.h"         // fmt::format
#include "pybind11/cast.h"      // py::arg
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kw_only, py::object, py::attribute_error, py::return_value_policy
#include "pybind11/pytypes.h"   // py::type
#include "pybind11/stl.h"       // NOLINT: support for STL types

#include <memory>    // std::make_unique
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::move
#include <variant>   // std::visit

namespace py = pybind11;

void init_regression_data_set(py::module_ &m) {
    using plssvm::bindings::python::util::regression_data_set_wrapper;

    py::class_<regression_data_set_wrapper>(m, "RegressionDataSet", "Encapsulate all necessary data that is needed for training or predicting using an C-SVR.")
        .def(py::init([](const std::string &filename, const std::optional<py::type> &type, const plssvm::file_format_type format, const std::optional<plssvm::min_max_scaler> &scaler, plssvm::mpi::communicator comm) {
                 if (type.has_value()) {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(comm), filename, format, scaler.value()));
                     }
                     return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(comm), filename, format));
                 }
                 if (scaler.has_value()) {
                     return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(comm), filename, format, scaler.value() });
                 }
                 return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(comm), filename, format });
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("filename"),
             py::kw_only(),
             py::arg("type") = std::nullopt,
             py::arg("format") = plssvm::file_format_type::libsvm,
             py::arg("scaler") = std::nullopt,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](plssvm::soa_matrix<plssvm::real_type> data, const std::optional<py::type> &type, const std::optional<plssvm::min_max_scaler> &scaler, plssvm::mpi::communicator comm) {
                 if (type.has_value()) {
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(comm), std::move(data), scaler.value()));
                     }
                     return std::make_unique<regression_data_set_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_data_set, typename regression_data_set_wrapper::possible_data_set_types>(type.value(), std::move(comm), std::move(data)));
                 }
                 if (scaler.has_value()) {
                     return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(comm), std::move(data), scaler.value() });
                 }
                 return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<double>{ std::move(comm), std::move(data) });
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("X"),
             py::kw_only(),
             py::arg("type") = std::nullopt,
             py::arg("scaler") = std::nullopt,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](plssvm::soa_matrix<plssvm::real_type> data, plssvm::bindings::python::util::label_vector_wrapper<typename regression_data_set_wrapper::possible_vector_types> labels, const std::optional<plssvm::min_max_scaler> scaler, plssvm::mpi::communicator comm) {
                 return std::visit([&](auto &&labels_vector) {
                     using label_type = typename plssvm::detail::remove_cvref_t<decltype(labels_vector)>::value_type;
                     if (scaler.has_value()) {
                         return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<label_type>(std::move(comm), std::move(data), std::forward<decltype(labels_vector)>(labels_vector), scaler.value()));
                     }
                     return std::make_unique<regression_data_set_wrapper>(plssvm::regression_data_set<label_type>(std::move(comm), std::move(data), std::forward<decltype(labels_vector)>(labels_vector)));
                 },
                                   labels.labels);
             }),
             "create a new data set from the provided file and additional optional parameters",
             py::arg("X"),
             py::arg("y"),
             py::kw_only(),
             py::arg("scaler") = std::nullopt,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def("save", [](const regression_data_set_wrapper &self, const std::string &filename, const plssvm::file_format_type format) { std::visit([&filename, format](auto &&data) { data.save(filename, format); }, self.data_set); }, "save the data set to a file using the provided file format type", py::arg("filename"), py::kw_only(), py::arg("format") = plssvm::file_format_type::libsvm)
        .def("data", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return py::cast(data.data()); }, self.data_set); }, "the data saved as 2D vector")
        .def("has_labels", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.has_labels(); }, self.data_set); }, "check whether the data set has labels")
        // clang-format off
        .def("labels", [](const regression_data_set_wrapper &self) {
            return std::visit([](auto &&data) {
                if (!data.has_labels()) {
                    throw py::attribute_error{ "'RegressionDataSet' object has no function 'labels'. Maybe this RegressionDataSet was created without labels?" };
                }
                return plssvm::bindings::python::util::vector_to_pyarray(data.labels()->get());
            }, self.data_set); }, "the labels")
        // clang-format on
        .def("num_data_points", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.num_data_points(); }, self.data_set); }, "the number of data points in the data set")
        .def("num_features", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.num_features(); }, self.data_set); }, "the number of features per data point")
        .def("is_scaled", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.is_scaled(); }, self.data_set); }, "check whether the original data has been scaled to [lower, upper] bounds")
        .def("scaling_factors", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) {
            if (!data.is_scaled()) {
                throw py::attribute_error{ "'RegressionDataSet' object has no function 'scaling_factors'. Maybe this RegressionDataSet has not been scaled?" };
            }
            return data.scaling_factors().value(); }, self.data_set); }, py::return_value_policy::reference_internal, "the factors used to scale this data set")
        // clang-format off
        .def("communicator", [](const regression_data_set_wrapper &self) { return std::visit([](auto &&data) { return data.communicator(); }, self.data_set); }, "the associated MPI communicator")
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
