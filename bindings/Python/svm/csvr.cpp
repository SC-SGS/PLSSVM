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
#include "plssvm/detail/type_list.hpp"              // plssvm::detail::supported_label_types_regression
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter, named parameters
#include "plssvm/solver_types.hpp"                  // plssvm::solver_type

#include "bindings/Python/svm/utility.hpp"  // plssvm::bindings::python::util::assemble_csvm
#include "bindings/Python/utility.hpp"      // plssvm::bindings::python::util::{check_kwargs_for_correctness, vector_to_pyarray, instantiate_class_bindings}

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_

#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace py = pybind11;

/**
 * @brief Functor to instantiate all C-SVR bindings.
 * @tparam label_type the label type for the C-SVR
 */
template <typename label_type>
struct csvr_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] csvr the Python C-SVR class
     */
    void operator()(py::class_<plssvm::csvr> &csvr, label_type) {
        csvr.def(
                "fit", [](const plssvm::csvr &self, const plssvm::regression_data_set<label_type> &data, const py::kwargs &args) {
                    // check keyword arguments
                    plssvm::bindings::python::util::check_kwargs_for_correctness(args, { "epsilon", "max_iter", "solver" });

                    auto epsilon{ plssvm::real_type{ 0.001 } };
                    if (args.contains("epsilon")) {
                        epsilon = args["epsilon"].cast<plssvm::real_type>();
                    }

                    plssvm::solver_type solver{ plssvm::solver_type::automatic };
                    if (args.contains("solver")) {
                        solver = args["solver"].cast<plssvm::solver_type>();
                    }

                    if (args.contains("max_iter")) {
                        return self.fit(data,
                                        plssvm::epsilon = epsilon,
                                        plssvm::max_iter = args["max_iter"].cast<unsigned long long>(),
                                        plssvm::solver = solver);
                    } else {
                        return self.fit(data,
                                        plssvm::epsilon = epsilon,
                                        plssvm::solver = solver);
                    }
                },
                "fit a model using the current SVM on the provided data")
            .def("predict", [](const plssvm::csvr &self, const plssvm::regression_model<label_type> &model, const plssvm::regression_data_set<label_type> &data) {
                if constexpr (std::is_same_v<label_type, std::string>) {
                    return self.predict<label_type>(model, data);
                } else {
                    return plssvm::bindings::python::util::vector_to_pyarray(self.predict<label_type>(model, data));
                } }, "predict the labels for a data set using a previously learned model")
            .def("score", py::overload_cast<const plssvm::regression_model<label_type> &>(&plssvm::csvr::score<label_type>, py::const_), "calculate the accuracy of the model")
            .def("score", py::overload_cast<const plssvm::regression_model<label_type> &, const plssvm::regression_data_set<label_type> &>(&plssvm::csvr::score<label_type>, py::const_), "calculate the accuracy of a data set using the model");
    }
};

void init_csvr(py::module_ &m, py::module_ &pure_virtual) {
    py::class_<plssvm::csvr> py_csvr(pure_virtual, "__pure_virtual_base_CSVR");

    // instantiate all functions using all available label_type
    plssvm::bindings::python::util::instantiate_class_bindings<csvr_bindings, plssvm::detail::supported_label_types_regression>(py_csvr);

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
             "create an C-SVR with the provided parameters and keyword arguments; the values in params will be overwritten by the keyword arguments");
}
