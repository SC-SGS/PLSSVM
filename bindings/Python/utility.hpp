#ifndef PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "fmt/format.h"         // std::format
#include "pybind11/pybind11.h"  // py::kwargs, py::value_error
#include "pybind11/stl.h"       // support for STL types

#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace py = pybind11;

inline void check_kwargs_for_correctness(py::kwargs args, const std::vector<std::string_view> valid_named_args) {
    for (const auto &[key, value] : args) {
        if (!plssvm::detail::contains(valid_named_args, key.cast<std::string_view>())) {
            throw py::value_error(fmt::format("Invalid argument \"{}={}\" provided!", key.cast<std::string_view>(), value.cast<std::string_view>()));
        }
    }
}

inline plssvm::parameter convert_kwargs_to_parameter(py::kwargs args, plssvm::parameter params = {}) {
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

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
