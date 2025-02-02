/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm/csvr.hpp"  // plssvm::csvr

#include "plssvm/constants.hpp"                     // plssvm::real_type
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter, named parameters
#include "plssvm/solver_types.hpp"                  // plssvm::solver_type

#include "bindings/Python/data_set/variant_wrapper.hpp"  // plssvm::bindings::python::util::regression_data_set_wrapper
#include "bindings/Python/model/variant_wrapper.hpp"     // plssvm::bindings::python::util::regression_model_wrapper
#include "bindings/Python/svm/utility.hpp"               // plssvm::bindings::python::util::assemble_csvm
#include "bindings/Python/utility.hpp"                   // plssvm::bindings::python::util::{check_kwargs_for_correctness, python_type_name_mapping, vector_to_pyarray}

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::module_, py::class_, py::init, py::kwargs, py::value_error
#include "pybind11/stl.h"       // support for STL types: std::vector

#include <exception>    // std::exception
#include <string_view>  // std::string_view
#include <variant>      // std::visit, std::get

namespace py = pybind11;

void init_csvr(py::module_ &m, py::module_ &pure_virtual) {
    using plssvm::bindings::python::util::regression_data_set_wrapper;
    using plssvm::bindings::python::util::regression_model_wrapper;

    const py::class_<plssvm::csvr> py_csvr(pure_virtual, "__pure_virtual_base_CSVR");

    // bind plssvm::make_csvm factory functions to "generic" Python C-SVR class
    py::class_<plssvm::csvr>(m, "CSVR", py_csvr, py::module_local())
        // IMPLICIT BACKEND
        .def(py::init([](const py::kwargs &args) {
                 return plssvm::bindings::python::util::assemble_csvm<plssvm::csvr>(args);
             }),
             "create an C-SVR with the provided keyword arguments")
        .def(py::init([](const plssvm::parameter &params, const py::kwargs &args) {
                 return plssvm::bindings::python::util::assemble_csvm<plssvm::csvr>(args, params);
             }),
             "create an C-SVR with the provided parameters and keyword arguments; the values in params will be overwritten by the keyword arguments")
        // clang-format off
        .def("fit", [](const plssvm::csvr &self, const regression_data_set_wrapper &data_set, const py::kwargs &args) {
                return std::visit([&](auto &&data) {
                    // check keyword arguments
                    plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "epsilon", "max_iter", "solver" });

                    auto epsilon{ plssvm::real_type{ 1e-10 } };
                    if (args.contains("epsilon")) {
                        epsilon = args["epsilon"].cast<plssvm::real_type>();
                    }

                    plssvm::solver_type solver{ plssvm::solver_type::automatic };
                    if (args.contains("solver")) {
                        solver = args["solver"].cast<plssvm::solver_type>();
                    }

                    if (args.contains("max_iter")) {
                        return regression_model_wrapper{ self.fit(data,
                                                                  plssvm::epsilon = epsilon,
                                                                  plssvm::max_iter = args["max_iter"].cast<unsigned long long>(),
                                                                  plssvm::solver = solver) };
                    } else {
                        return regression_model_wrapper{ self.fit(data,
                                                                  plssvm::epsilon = epsilon,
                                                                  plssvm::solver = solver) };
                    }
                }, data_set.data_set); }, "fit a model using the current SVM on the provided data")
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
                }, trained_model.model); }, "predict the labels for a data set using a previously learned model")
        .def("score", [](const plssvm::csvr &self, const regression_model_wrapper &trained_model) {
                return std::visit([&](auto &&model) {
                    return self.score(model);
                }, trained_model.model); }, "calculate the accuracy of the model")
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
                    }, trained_model.model); }, "calculate the accuracy of the model");
    // clang-format on
}
