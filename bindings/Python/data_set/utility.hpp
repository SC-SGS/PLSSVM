/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions used for creating the Pybind11 Python bindings for the data set classes.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_DATA_SET_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_DATA_SET_UTILITY_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "pybind11/pybind11.h"  // py::kwargs, py::cast_error, py::attribute_error

#include <array>  // std::array

namespace py = pybind11;

namespace plssvm::bindings::python::util {

/**
 * @brief Create the necessary PLSSVM scaling objects to initialize a data set.
 * @tparam data_set_type the type of the data set
 * @param[in] args the Python arguments used to initialize the scaling object
 * @return the constructed scaling object (`[[nodiscard]]`)
 */
template <typename data_set_type>
[[nodiscard]] inline typename data_set_type::scaling create_scaling_object(const py::kwargs &args) {
    if (args.contains("scaling")) {
        typename data_set_type::scaling scaling{ plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } };

        // try to directly convert it to a plssvm::data_set_type::scaling object
        try {
            scaling = args["scaling"].cast<typename data_set_type::scaling>();
        } catch (const py::cast_error &) {
            // can't cast to plssvm::data_set_type::scaling
            // -> try a std::array<real_type, 2> instead!
            try {
                const auto interval = args["scaling"].cast<std::array<plssvm::real_type, 2>>();
                scaling = typename data_set_type::scaling{ interval[0], interval[1] };
            } catch (...) {
                // rethrow exception if this also did not succeed
                throw;
            }
        }
        return scaling;
    } else {
        throw py::attribute_error{ "Can't extract scaling information, no scaling keyword argument given!" };
    }
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_DATA_SET_UTILITY_HPP_
