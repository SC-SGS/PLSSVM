/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm/csvc.hpp"  // plssvm::csvc

#include "plssvm/classification_types.hpp"              // plssvm::classification_type
#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/detail/type_list.hpp"                  // plssvm::detail::supported_label_types_classification
#include "plssvm/model/classification_model.hpp"        // plssvm::classification_model
#include "plssvm/parameter.hpp"                         // plssvm::parameter, named parameters
#include "plssvm/solver_types.hpp"                      // plssvm::solver_type

#include "bindings/Python/svm/utility.hpp"  // assemble_csvm
#include "bindings/Python/utility.hpp"      // check_kwargs_for_correctness

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_

#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace py = pybind11;

/**
 * @brief Functor to instantiate all CSVC bindings.
 * @tparam label_type the label type for the CSVC
 */
template <typename label_type>
struct csvc_bindings {
    /**
     * @brief Function call operator to initialize the Python bindings.
     * @param[in] csvc the Python CSVR class
     */
    void operator()(py::class_<plssvm::csvc> &csvc, label_type) {
        csvc.def(
                "fit", [](const plssvm::csvc &self, const plssvm::classification_data_set<label_type> &data, const py::kwargs &args) {
                    // check keyword arguments
                    check_kwargs_for_correctness(args, { "epsilon", "max_iter", "classification", "solver" });

                    auto epsilon{ plssvm::real_type{ 0.001 } };
                    if (args.contains("epsilon")) {
                        epsilon = args["epsilon"].cast<plssvm::real_type>();
                    }

                    // can't do it with max_iter due to OAO splitting the data set

                    plssvm::classification_type classification{ plssvm::classification_type::oaa };
                    if (args.contains("classification")) {
                        classification = args["classification"].cast<plssvm::classification_type>();
                    }

                    plssvm::solver_type solver{ plssvm::solver_type::automatic };
                    if (args.contains("solver")) {
                        solver = args["solver"].cast<plssvm::solver_type>();
                    }

                    if (args.contains("max_iter")) {
                        return self.fit(data,
                                        plssvm::epsilon = epsilon,
                                        plssvm::max_iter = args["max_iter"].cast<unsigned long long>(),
                                        plssvm::classification = classification,
                                        plssvm::solver = solver);
                    } else {
                        return self.fit(data,
                                        plssvm::epsilon = epsilon,
                                        plssvm::classification = classification,
                                        plssvm::solver = solver);
                    }
                },
                "fit a model using the current SVM on the provided data")
            .def("predict", [](const plssvm::csvc &self, const plssvm::classification_model<label_type> &model, const plssvm::classification_data_set<label_type> &data) {
                   if constexpr (std::is_same_v<label_type, std::string>) {
                       return self.predict<label_type>(model, data);
                   } else {
                       return vector_to_pyarray(self.predict<label_type>(model, data));
                   } }, "predict the labels for a data set using a previously learned model")
            .def("score", py::overload_cast<const plssvm::classification_model<label_type> &>(&plssvm::csvc::score<label_type>, py::const_), "calculate the accuracy of the model")
            .def("score", py::overload_cast<const plssvm::classification_model<label_type> &, const plssvm::classification_data_set<label_type> &>(&plssvm::csvc::score<label_type>, py::const_), "calculate the accuracy of a data set using the model");
    }
};

void init_csvc(py::module_ &m, py::module_ &pure_virtual) {
    py::class_<plssvm::csvc> py_csvc(pure_virtual, "__pure_virtual_base_CSVC");

    // instantiate all functions using all available label_type
    instantiate_csvm<csvc_bindings, plssvm::detail::supported_label_types_classification>(py_csvc);

    // bind plssvm::make_csvm factory functions to "generic" Python CSVC class
    py::class_<plssvm::csvc>(m, "CSVC", py_csvc, py::module_local())
        // IMPLICIT BACKEND
        .def(py::init([](const py::kwargs &args) {
                 return assemble_csvm<plssvm::csvc>(args);
             }),
             "create an CSVC with the provided keyword arguments")
        .def(py::init([](const plssvm::parameter &params, const py::kwargs &args) {
                 return assemble_csvm<plssvm::csvc>(args, params);
             }),
             "create an CSVC with the provided parameters and keyword arguments; the values in params will be overwritten by the keyword arguments");
}
