#ifndef PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "fmt/format.h"         // std::format
#include "pybind11/pybind11.h"  // py::kwargs, py::value_error, py::exception
#include "pybind11/stl.h"       // support for STL types

#include <exception>    // std::exception_ptr, std::rethrow_exception
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace py = pybind11;

/**
 * @brief Check that the Python kwargs @p args only contain keyword arguments with names present in @p valid_named_args.
 * @param[in] args the Python keyword arguments
 * @param[in] valid_named_args the valid keyword arguments
 * @throws pybind11::value_error if an illegal keyword arguments has been provided
 */
inline void check_kwargs_for_correctness(py::kwargs args, const std::vector<std::string_view> valid_named_args) {
    for (const auto &[key, value] : args) {
        if (!plssvm::detail::contains(valid_named_args, key.cast<std::string_view>())) {
            throw py::value_error(fmt::format("Invalid argument \"{}={}\" provided!", key.cast<std::string_view>(), value.cast<std::string_view>()));
        }
    }
}

/**
 * @brief Convert the Python kwargs @p args to an `plssvm::parameter` object.
 * @param[in] args the Python keyword arguments
 * @param[in] params the baseline parameter
 * @return the `plssvm::parameter` object filled with the keyword @p args (`[[nodiscard]]`)
 */
[[nodiscard]] inline plssvm::parameter convert_kwargs_to_parameter(py::kwargs args, plssvm::parameter params = {}) {
    if (args.contains("kernel_type")) {
        params.kernel_type = args["kernel_type"].cast<typename decltype(params.kernel_type)::value_type>();
    }
    if (args.contains("degree")) {
        params.degree = args["degree"].cast<typename decltype(params.degree)::value_type>();
    }
    if (args.contains("gamma")) {
        params.gamma = args["gamma"].cast<typename decltype(params.gamma)::value_type>();
    }
    if (args.contains("coef0")) {
        params.coef0 = args["coef0"].cast<typename decltype(params.coef0)::value_type>();
    }
    if (args.contains("cost")) {
        params.cost = args["cost"].cast<typename decltype(params.cost)::value_type>();
    }
    return params;
}

/**
 * @def PLSSVM_REGISTER_EXCEPTION
 * @brief Register the PLSSVM exception @p exception_type derived from @p base_exception in the Python module
 *        @p py_module using the Python name @p py_exception_name.
 */
#define PLSSVM_REGISTER_EXCEPTION(exception_type, py_module, py_exception_name, base_exception)                  \
    static py::exception<exception_type> py_exception_name(py_module, #py_exception_name, base_exception.ptr()); \
    py::register_exception_translator([](std::exception_ptr p) {                                                 \
        try {                                                                                                    \
            if (p) {                                                                                             \
                std::rethrow_exception(p);                                                                       \
            }                                                                                                    \
        } catch (const exception_type &e) {                                                                      \
            py_exception_name(e.what_with_loc().c_str());                                                        \
        }                                                                                                        \
    });

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
