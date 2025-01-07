/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions used for creating the Pybind11 Python bindings for the CSVM classes.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                          // plssvm::backend_type, plssvm::determine_default_backend, plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_types.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_types.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/csvm_factory.hpp"                           // plssvm::make_csvm
#include "plssvm/parameter.hpp"                              // plssvm::parameter, named parameters
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform, plssvm::determine_default_target_platform, plssvm::list_available_target_platforms

#include "bindings/Python/utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter

#include "pybind11/pybind11.h"  // py::class_, py::kwargs, py::cast_error, py::attribute_error, py::value_error

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <memory>   // std::unique_ptr
#include <sstream>  // std::istringstream
#include <string>   // std::string
#include <tuple>    // std::tuple_element_t, std::tuple_size_v
#include <utility>  // std::integer_sequence, std::make_integer_sequence

namespace py = pybind11;

/**
 * @brief Assemble a CSVM (CSVC or CSVR based on the template parameter @p csvm_type) using the named Python arguments @p args and PLSSVM parameters @p input_params.
 * @tparam csvm_type the type of the CSVM to create
 * @param[in] args the named Python arguments
 * @param[in] input_params the PLSSVM parameter
 * @return the created CSVM (`[[nodiscard]]`)
 */
template <typename csvm_type>
[[nodiscard]] inline std::unique_ptr<csvm_type> assemble_csvm(const py::kwargs &args, plssvm::parameter input_params = {}) {
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

        return plssvm::make_csvm<csvm_type>(backend, target, params, plssvm::sycl_implementation_type = impl_type, plssvm::sycl_kernel_invocation_type = invocation_type);
    } else {
        return plssvm::make_csvm<csvm_type>(backend, target, params);
    }
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @tparam Idx the label type indices
 * @param[in] csvm the Python CSVM used for instantiation
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, typename csvm_type, std::size_t... Idx>
void instantiate_csvm(py::class_<csvm_type> &csvm, std::integer_sequence<std::size_t, Idx...>) {
    (InstantiationFunction<std::tuple_element_t<Idx, LabelTypes>>{}(csvm, std::tuple_element_t<Idx, LabelTypes>{}), ...);
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @param[in] csvm the Python CSVM used for instantiation
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, typename csvm_type>
void instantiate_csvm(py::class_<csvm_type> &csvm) {
    instantiate_csvm<InstantiationFunction, LabelTypes, csvm_type>(csvm, std::make_integer_sequence<std::size_t, std::tuple_size_v<LabelTypes>>{});
}

#endif  // PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
