/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"

#include "plssvm/backends/SYCL/implementation_type.hpp"
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"
#include "plssvm/constants.hpp"  // plssvm::real_type
#include "plssvm/csvm_factory.hpp"

#include "utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <tuple>        // std::tuple_element_t, std::tuple_size_v
#include <type_traits>  // std::is_same_v
#include <utility>      // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

template <typename label_type>
void instantiate_csvm_functions(py::class_<plssvm::csvm> &c, label_type) {
    c.def(
         "fit", [](const plssvm::csvm &self, const plssvm::data_set<label_type> &data, const py::kwargs &args) {
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
        .def(
            "predict", [](const plssvm::csvm &self, const plssvm::model<label_type> &model, const plssvm::data_set<label_type> &data) {
                if constexpr (std::is_same_v<label_type, std::string>) {
                    return self.predict<label_type>(model, data);
                } else {
                    return vector_to_pyarray(self.predict<label_type>(model, data));
                }
            },
            "predict the labels for a data set using a previously learned model")
        .def("score", py::overload_cast<const plssvm::model<label_type> &>(&plssvm::csvm::score<label_type>, py::const_), "calculate the accuracy of the model")
        .def("score", py::overload_cast<const plssvm::model<label_type> &, const plssvm::data_set<label_type> &>(&plssvm::csvm::score<label_type>, py::const_), "calculate the accuracy of a data set using the model");
}

template <typename T, std::size_t... Idx>
void instantiate_csvm_functions(py::class_<plssvm::csvm> &c, std::integer_sequence<std::size_t, Idx...>) {
    (instantiate_csvm_functions(c, std::tuple_element_t<Idx, T>{}), ...);
}

template <typename T>
void instantiate_model_bindings(py::class_<plssvm::csvm> &c) {
    instantiate_csvm_functions<T>(c, std::make_integer_sequence<std::size_t, std::tuple_size_v<T>>{});
}

std::unique_ptr<plssvm::csvm> assemble_csvm(const py::kwargs &args, plssvm::parameter input_params = {}) {
    // check keyword arguments
    check_kwargs_for_correctness(args, { "backend", "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_implementation_type", "sycl_kernel_invocation_type" });
    // if one of the value keyword parameter is provided, set the respective value
    const plssvm::parameter params = convert_kwargs_to_parameter(args, input_params);
    plssvm::backend_type backend = plssvm::determine_default_backend();
    if (args.contains("backend")) {
        if (py::isinstance<py::str>(args["backend"])) {
            std::istringstream iss{ args["backend"].cast<std::string>() };
            iss >> backend;
            if (iss.fail()) {
                throw py::value_error{ fmt::format("Available backends are \"{}\", got {}!", fmt::join(plssvm::list_available_backends(), ";"), args["backend"].cast<std::string>()) };
            }
        } else {
            backend = args["backend"].cast<plssvm::backend_type>();
        }
    }
    plssvm::target_platform target = plssvm::determine_default_target_platform();
    if (args.contains("target_platform")) {
        if (py::isinstance<py::str>(args["target_platform"])) {
            std::istringstream iss{ args["target_platform"].cast<std::string>() };
            iss >> target;
            if (iss.fail()) {
                throw py::value_error{ fmt::format("Available target platforms are \"{}\", got {}!", fmt::join(plssvm::list_available_target_platforms(), ";"), args["target_platform"].cast<std::string>()) };
            }
        } else {
            target = args["target_platform"].cast<plssvm::target_platform>();
        }
    }

    // parse SYCL specific keyword arguments
    if (backend == plssvm::backend_type::sycl) {
        // sycl specific flags
        plssvm::sycl::implementation_type impl_type = plssvm::sycl::implementation_type::automatic;
        if (args.contains("sycl_implementation_type")) {
            impl_type = args["sycl_implementation_type"].cast<plssvm::sycl::implementation_type>();
        }
        plssvm::sycl::kernel_invocation_type invocation_type = plssvm::sycl::kernel_invocation_type::automatic;
        if (args.contains("sycl_kernel_invocation_type")) {
            invocation_type = args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>();
        }

        return plssvm::make_csvm(backend, target, params, plssvm::sycl_implementation_type = impl_type, plssvm::sycl_kernel_invocation_type = invocation_type);
    } else {
        return plssvm::make_csvm(backend, target, params);
    }
}

void init_csvm(py::module_ &m) {
    const py::module_ pure_virtual_model = m.def_submodule("__pure_virtual");

    py::class_<plssvm::csvm> pycsvm(pure_virtual_model, "__pure_virtual_base_CSVM");
    pycsvm.def("get_params", &plssvm::csvm::get_params, "get the parameter used for this SVM")
        .def(
            "set_params", [](plssvm::csvm &self, const plssvm::parameter &params) {
                self.set_params(params);
            },
            "update the parameter used for this SVM using a plssvm.Parameter object")
        .def(
            "set_params", [](plssvm::csvm &self, const py::kwargs &args) {
                // check keyword arguments
                check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
                // convert kwargs to parameter and update csvm internal parameter
                self.set_params(convert_kwargs_to_parameter(args, self.get_params()));
            },
            "update the parameter used for this SVM using keyword arguments")
        .def("get_target_platform", &plssvm::csvm::get_target_platform, "get the actual target platform this SVM runs on");

    // instantiate all functions using all available label_type
    instantiate_model_bindings<plssvm::detail::supported_label_types>(pycsvm);

    // bind plssvm::make_csvm factory function to a "generic" Python csvm class
    py::class_<plssvm::csvm>(m, "CSVM", pycsvm, py::module_local())
        // IMPLICIT BACKEND
        .def(py::init([](const py::kwargs &args) {
                 return assemble_csvm(args);
             }),
             "create an SVM with the provided keyword arguments")
        .def(py::init([](const plssvm::parameter &params, const py::kwargs &args) {
                 return assemble_csvm(args, params);
             }),
             "create an SVM with the provided parameters and keyword arguments; the values in params will be overwritten by the keyword arguments");
}