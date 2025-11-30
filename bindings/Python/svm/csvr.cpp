/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm/csvr.hpp"  // plssvm::csvr

#include "plssvm/backend_types.hpp"                 // plssvm::backend_type
#include "plssvm/constants.hpp"                     // plssvm::real_type, plssvm::DEFAULT_EPSILON
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/detail/type_traits.hpp"            // plssvm::detail::remove_cvref_t
#include "plssvm/gamma.hpp"                         // plssvm::gamma_type
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"              // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                     // plssvm::parameter, named arguments
#include "plssvm/solver_types.hpp"                  // plssvm::solver_type
#include "plssvm/target_platforms.hpp"              // plssvm::target_platform

#include "bindings/Python/bindings_fwd.hpp"                 // forward declare all helper functions to create the Python bindings
#include "bindings/Python/data_set/variant_wrapper.hpp"     // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/model/variant_wrapper.hpp"        // plssvm::bindings::python::util::regression_model_wrapper
#include "bindings/Python/svm/utility.hpp"                  // plssvm::bindings::python::util::assemble_csvm
#include "bindings/Python/type_caster/mpi_type_caster.hpp"  // NOLINT: a custom Pybind11 type caster for a plssvm::mpi::communicator
#include "bindings/Python/utility.hpp"                      // plssvm::bindings::python::util::{python_type_name_mapping, vector_to_pyarray}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::arg, py::kw_only, py::value_error
#include "pybind11/pytypes.h"   // py::kwargs
#include "pybind11/stl.h"       // NOLINT: support for STL types: std::optional

#include <exception>    // std::exception
#include <optional>     // std::optional, std::nullopt
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <variant>      // std::visit, std::get

namespace py = pybind11;

void init_csvr(py::module_ &m) {
    using plssvm::bindings::python::util::regression_data_set_wrapper;
    using plssvm::bindings::python::util::regression_model_wrapper;

    // the default parameters used
    const plssvm::parameter default_params{};

    // bind plssvm::make_csvm factory functions to "generic" Python C-SVR class
    py::class_<plssvm::csvr, plssvm::csvm>(m, "CSVR", "Base class for all backend C-SVR implementations.")
        // IMPLICIT BACKEND
        .def(py::init([](const plssvm::backend_type backend, const plssvm::target_platform target, const plssvm::parameter &params, plssvm::mpi::communicator comm, const py::kwargs &optional_args) {
                 return plssvm::bindings::python::util::assemble_csvm<plssvm::csvr>(backend, target, params, std::move(comm), optional_args);
             }),
             "create an C-SVR with the provided SVM parameter encapsulated in a plssvm.Parameter",
             py::arg("backend") = plssvm::backend_type::automatic,
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("params") = default_params,
             py::arg("comm") = plssvm::mpi::communicator{})
        .def(py::init([](const plssvm::backend_type backend, const plssvm::target_platform target, const plssvm::kernel_function_type kernel_type, const int degree, const plssvm::gamma_type gamma, const plssvm::real_type coef0, const plssvm::real_type cost, plssvm::mpi::communicator comm, const py::kwargs &optional_args) {
                 const plssvm::parameter params{ kernel_type, degree, gamma, coef0, cost };
                 return plssvm::bindings::python::util::assemble_csvm<plssvm::csvr>(backend, target, params, std::move(comm), optional_args);
             }),
             "create an C-SVR with the provided SVM parameter as separate keyword arguments",
             py::arg("backend") = plssvm::backend_type::automatic,
             py::arg("target") = plssvm::target_platform::automatic,
             py::kw_only(),
             py::arg("kernel_type") = default_params.kernel_type,
             py::arg("degree") = default_params.degree,
             py::arg("gamma") = default_params.gamma,
             py::arg("coef0") = default_params.coef0,
             py::arg("cost") = default_params.cost,
             py::arg("comm") = plssvm::mpi::communicator{})
        // clang-format off
        .def("fit", [](const plssvm::csvr &self, const regression_data_set_wrapper &data_set, const plssvm::real_type epsilon, const std::optional<unsigned long long> max_iter, const plssvm::solver_type solver) {
                return std::visit([&](auto &&data) {
                    if (max_iter.has_value()) {
                        return regression_model_wrapper{ self.fit(data,
                                                                      plssvm::epsilon = epsilon,
                                                                      plssvm::max_iter = max_iter.value(),
                                                                      plssvm::solver = solver) };
                    }
                    return regression_model_wrapper{ self.fit(data,
                                                                  plssvm::epsilon = epsilon,
                                                                  plssvm::solver = solver) };
                }, data_set.data_set); }, "fit a model using the current C-SVR on the provided data",
                py::arg("data"),
                py::kw_only(),
                py::arg("epsilon") = plssvm::real_type{ 1e-10 },
                py::arg("max_iter") = std::nullopt,
                py::arg("solver") = plssvm::solver_type::automatic)
        .def("predict", [](const plssvm::csvr &self, const regression_model_wrapper &trained_model, const regression_data_set_wrapper &data_set) {
                return std::visit([&](auto &&model) {
                    using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                    try {
                        return plssvm::bindings::python::util::vector_to_pyarray(self.predict(model, std::get<plssvm::regression_data_set<label_type>>(data_set.data_set)));
                    } catch (const std::exception &) {
                        using plssvm::bindings::python::util::python_type_name_mapping;
                        const std::string_view data_set_label_type = std::visit([](auto &&data) {
                            return python_type_name_mapping<typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type>();
                        }, data_set.data_set);
                        throw py::value_error{ fmt::format("Mismatching label types! Trained the model with {}, but tried to predict it with {}.", python_type_name_mapping<label_type>(), data_set_label_type) };
                    }
                }, trained_model.model); }, "predict the labels for a data set using a previously learned model", py::arg("model"), py::arg("data"))
        .def("score", [](const plssvm::csvr &self, const regression_model_wrapper &trained_model) {
                return std::visit([&](auto &&model) {
                    return self.score(model);
                }, trained_model.model); }, "calculate the accuracy of the model", py::arg("model"))
        .def("score", [](const plssvm::csvr &self, const regression_model_wrapper &trained_model, const regression_data_set_wrapper &data_set) {
                return std::visit([&](auto &&model) {
                    using label_type = typename plssvm::detail::remove_cvref_t<decltype(model)>::label_type;
                    try {
                        return self.score(model, std::get<plssvm::regression_data_set<label_type>>(data_set.data_set));
                    } catch (const std::exception &) {
                        using plssvm::bindings::python::util::python_type_name_mapping;
                        const std::string_view data_set_label_type = std::visit([](auto &&data) {
                            return python_type_name_mapping<typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type>();
                        }, data_set.data_set);
                        throw py::value_error{ fmt::format("Mismatching label types! Trained the model with {}, but tried to score it with {}.", python_type_name_mapping<label_type>(), data_set_label_type) };
                    }
                    }, trained_model.model); }, "calculate the accuracy of the model", py::arg("model"), py::arg("data"));
    // clang-format on
}
