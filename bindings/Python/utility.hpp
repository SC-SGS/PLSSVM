#ifndef PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

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

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
