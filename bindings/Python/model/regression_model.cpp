/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/model/regression_model.hpp"  // plssvm::regression_model

#include "plssvm/constants.hpp"           // plssvm::real_type
#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t
#include "plssvm/matrix.hpp"              // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"    // plssvm::mpi::communicator

#include "bindings/Python/bindings_fwd.hpp"                 // forward declare all helper functions to create the Python bindings
#include "bindings/Python/model/variant_wrapper.hpp"        // plssvm::bindings::python::util::regression_model_wrapper
#include "bindings/Python/type_caster/mpi_type_caster.hpp"  // a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                      // plssvm::bindings::python::util::{python_type_name_mapping, create_instance, vector_to_pyarray}

#include "fmt/format.h"         // fmt::format
#include "fmt/ranges.h"         // fmt::join
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::kw_only, py::array, py::list
#include "pybind11/pytypes.h"   // py::type
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <memory>    // std::make_unique
#include <optional>  // std::optional, std::make_optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::move
#include <variant>   // std::visit

namespace py = pybind11;

void init_regression_model(py::module_ &m) {
    using plssvm::bindings::python::util::regression_model_wrapper;

    py::class_<regression_model_wrapper>(m, "RegressionModel", "Implements a class encapsulating the result of a call to the C-SVR fit function. A model is used to predict the labels of a new data set.")
        .def(py::init([](const std::string &filename, const std::optional<py::type> type, plssvm::mpi::communicator comm) {
                 if (type.has_value()) {
                     return std::make_unique<regression_model_wrapper>(plssvm::bindings::python::util::create_instance<plssvm::regression_model, typename regression_model_wrapper::possible_model_types>(type.value(), std::move(comm), filename));
                 } else {
                     return std::make_unique<regression_model_wrapper>(plssvm::regression_model<double>{ std::move(comm), filename });
                 }
             }),
             "load a previously learned regression model from a file",
             py::arg("filename"),
             py::kw_only(),
             py::arg("type") = std::nullopt,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def("save", [](const regression_model_wrapper &self, const std::string &filename) { return std::visit([&filename](auto &&model) { model.save(filename); }, self.model); }, "save the current model to a file", py::arg("filename"))
        .def("num_support_vectors", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return model.num_support_vectors(); }, self.model); }, "the number of support vectors (note: all training points become support vectors for LS-SVMs)")
        .def("num_features", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return model.num_features(); }, self.model); }, "the number of features of the support vectors")
        .def("get_params", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return model.get_params(); }, self.model); }, "the C-SVR hyper-parameters used to learn this model")
        .def("support_vectors", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return py::cast(model.support_vectors()); }, self.model); }, "the support vectors (note: all training points become support vectors for LS-SVMs)")
        // clang-format off
        .def("labels", [](const regression_model_wrapper &self) {
            return std::visit([](auto &&model) -> std::optional<py::array> {
                if (model.labels().has_value()) {
                    return std::make_optional(plssvm::bindings::python::util::vector_to_pyarray(model.labels()->get()));
                } else {
                    return std::nullopt;
                }
            }, self.model); }, "the labels")
        .def("weights", [](const regression_model_wrapper &self) {
            return std::visit([](auto &&model) {
                py::list ret{};
                for (const plssvm::aos_matrix<plssvm::real_type> &matr : model.weights()) {
                    ret.append(py::cast(matr));
                }
                return ret; },
            self.model); }, "the weights learned for each support vector")
        // clang-format on
        .def("rho", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return plssvm::bindings::python::util::vector_to_pyarray(model.rho()); }, self.model); }, "the bias value after learning")
        // clang-format off
        .def("communicator", [](const regression_model_wrapper &self) { return std::visit([](auto &&model) { return model.communicator(); }, self.model); }, "the associated MPI communicator")
        .def("__repr__", [](const regression_model_wrapper &self) {
            return std::visit([](auto &&model) {
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                return fmt::format("<plssvm.RegressionModel with {{ label_type: {}, #sv: {}, #features: {}, rho: [{}] }}>",
                    plssvm::bindings::python::util::python_type_name_mapping<label_type>(),
                    model.num_support_vectors(),
                    model.num_features(),
                    fmt::join(model.rho(), ","));
            }, self.model); });
    // clang-format on
}
