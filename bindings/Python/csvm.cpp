/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"
#include "plssvm/csvm_factory.hpp"

#include "utility.hpp"          // check_kwargs_for_correctness, convert_kwargs_to_parameter

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_
#include "pybind11/stl.h"       // support for STL types

#include <cstddef>              // std::size_t
#include <string>               // std::string
#include <tuple>                // std::tuple_element_t, std::tuple_size_v
#include <type_traits>          // std::is_same_v
#include <utility>              // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

template <typename real_type, typename label_type>
void instantiate_csvm_functions(py::class_<plssvm::csvm> &c, plssvm::detail::real_type_label_type_combination<real_type, label_type>) {
    c.def(
         "fit", [](const plssvm::csvm &self, const plssvm::data_set<real_type, label_type> &data, const py::kwargs &args) {
             // check keyword arguments
             check_kwargs_for_correctness(args, { "epsilon", "max_iter" });

             if (args.contains("epsilon") && args.contains("max_iter")) {
                 return self.fit(data, plssvm::epsilon = args["epsilon"].cast<real_type>(), plssvm::max_iter = args["max_iter"].cast<unsigned long long>());
             } else if (args.contains("epsilon")) {
                 return self.fit(data, plssvm::epsilon = args["epsilon"].cast<real_type>());
             } else if (args.contains("max_iter")) {
                 return self.fit(data, plssvm::max_iter = args["max_iter"].cast<unsigned long long>());
             } else {
                 return self.fit(data);
             }
         },
         "fit a model using the current SVM on the provided data")
        .def(
            "predict", [](const plssvm::csvm &self, const plssvm::model<real_type, label_type> &model, const plssvm::data_set<real_type, label_type> &data) {
                if constexpr (std::is_same_v<label_type, std::string>) {
                    return self.predict<real_type, label_type>(model, data);
                } else {
                    return vector_to_pyarray(self.predict<real_type, label_type>(model, data));
                }
            },
            "predict the labels for a data set using a previously learned model")
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_), "calculate the accuracy of the model")
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &, const plssvm::data_set<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_), "calculate the accuracy of a data set using the model");
}

template <typename T, std::size_t... Idx>
void instantiate_csvm_functions(py::class_<plssvm::csvm> &c, std::integer_sequence<std::size_t, Idx...>) {
    (instantiate_csvm_functions(c, std::tuple_element_t<Idx, T>{}), ...);
}

template <typename T>
void instantiate_model_bindings(py::class_<plssvm::csvm> &c) {
    instantiate_csvm_functions<T>(c, std::make_integer_sequence<std::size_t, std::tuple_size_v<T>>{});
}

std::unique_ptr<plssvm::csvm> assemble_csvm(const plssvm::backend_type backend, const plssvm::target_platform target, const py::kwargs &args) {
    // check keyword arguments
    check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_implementation_type", "sycl_kernel_invocation_type" });
    // if one of the value keyword parameter is provided, set the respective value
    const plssvm::parameter params = convert_kwargs_to_parameter(args);
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
            "update the parameter used for this SVM using keyword arguments");

    // instantiate all functions using all available real_type x label_type combinations
    instantiate_model_bindings<plssvm::detail::real_type_label_type_combination_list>(pycsvm);

    // bind plssvm::make_csvm factory function to a "generic" Python csvm class
    py::class_<plssvm::csvm>(m, "CSVM", pycsvm, py::module_local())
        // IMPLICIT BACKEND
        .def(py::init([]() {
                 return plssvm::make_csvm();
             }),
             "create an SVM with the default backend, default target platform and default parameters")
        .def(py::init([](const plssvm::parameter &params) {
                 return plssvm::make_csvm(params);
             }),
             "create an SVM with the default backend, default target platform and provided parameter object")
        .def(py::init([](const plssvm::target_platform target) {
                 return plssvm::make_csvm(target);
             }),
             "create an SVM with the default backend, provided target platform and default parameters")
        .def(py::init([](const plssvm::target_platform target, const plssvm::parameter &params) {
                 return plssvm::make_csvm(target, params);
             }),
             "create an SVM with the default backend, provided target platform and provided parameter object")
        .def(py::init([](const py::kwargs &args) {
                 // assemble CSVM
                 return assemble_csvm(plssvm::determine_default_backend(), plssvm::determine_default_target_platform(), args);
             }),
             "create an SVM with the default backend, default target platform and provided keyword arguments")
        .def(py::init([](const plssvm::target_platform target, const py::kwargs &args) {
                 // assemble CSVM
                 return assemble_csvm(plssvm::determine_default_backend(), target, args);
             }),
             "create an SVM with the default backend, provided target platform and provided keyword arguments")
        // EXPLICIT BACKEND
        .def(py::init([](const plssvm::backend_type backend) {
                 return plssvm::make_csvm(backend);
             }),
             "create an SVM with the provided backend, default target platform and default parameters")
        .def(py::init([](const plssvm::backend_type backend, const plssvm::parameter &params) {
                 return plssvm::make_csvm(backend, params);
             }),
             "create an SVM with the provided backend, default target platform and provided parameter object")
        .def(py::init([](const plssvm::backend_type backend, const plssvm::target_platform target) {
                 return plssvm::make_csvm(backend, target);
             }),
             "create an SVM with the provided backend, provided target platform and default parameters")
        .def(py::init([](const plssvm::backend_type backend, const plssvm::target_platform target, const plssvm::parameter &params) {
                 return plssvm::make_csvm(backend, target, params);
             }),
             "create an SVM with the provided backend, provided target platform and provided parameter object")
        .def(py::init([](const plssvm::backend_type backend, const py::kwargs &args) {
                 // assemble CSVM
                 return assemble_csvm(backend, plssvm::determine_default_target_platform(), args);
             }),
             "create an SVM with the provided backend, default target platform and provided keyword arguments")
        .def(py::init([](const plssvm::backend_type backend, const plssvm::target_platform target, const py::kwargs &args) {
                 // assemble CSVM
                 return assemble_csvm(backend, target, args);
             }),
             "create an SVM with the provided backend, provided target platform and provided keyword arguments");
}