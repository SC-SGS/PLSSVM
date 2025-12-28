/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom type caster for a matrix wrapper (storing a plssvm::matrix and optional feature names).
 */

#ifndef PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_WRAPPER_TYPE_CASTER_HPP_
#define PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_WRAPPER_TYPE_CASTER_HPP_
#pragma once

#include "plssvm/matrix.hpp"  // plssvm::matrix, plssvm::layout_type

#include "bindings/Python/type_caster/matrix_type_caster.hpp"  // custom plssvm::matrix type caster
#include "bindings/Python/utility.hpp"                         // plssvm::bindings::python::util::is_pandas_data_frame

#include "pybind11/cast.h"      // pybind11::detail::type_caster
#include "pybind11/pybind11.h"  // py::isinstance, py::value_error
#include "pybind11/pytypes.h"   // py::list, py::str

#include <optional>  // std::optional, std::make_optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::move
#include <vector>    // std::vector

namespace py = pybind11;

namespace plssvm::bindings::python::util {

/**
 * @brief A small wrapper around a plssvm::matrix that also allows us to store potential feature names.
 * @tparam T the value type of the PLSSVM matrix
 * @tparam layout the memory layout of the PLSSVM matrix
 */
template <typename T, plssvm::layout_type layout>
struct matrix_wrapper {
    /// The PLSSVM matrix.
    plssvm::matrix<T, layout> matrix{};
    /// The optionally available feature names.
    std::optional<std::vector<std::string>> feature_names;
};

/**
 * @brief Shortcut for a matrix_wrapper storing a plssvm::matrix in the Array-of-Structs layout.
 */
template <typename T>
using aos_matrix_wrapper = matrix_wrapper<T, plssvm::layout_type::aos>;

/**
 * @brief Shortcut for a matrix_wrapper storing a plssvm::matrix in the Struct-of-Arrays layout.
 */
template <typename T>
using soa_matrix_wrapper = matrix_wrapper<T, plssvm::layout_type::soa>;

}  // namespace plssvm::bindings::python::util

namespace pybind11::detail {

/**
 * @brief A custom Pybind11 type caster to convert Python object from and to a plssvm::bindings::python::util::matrix_wrapper.
 * @tparam T the value type of the PLSSVM matrix
 * @tparam layout the memory layout type of the PLSSVM matrix
 */
template <typename T, plssvm::layout_type layout>
struct type_caster<plssvm::bindings::python::util::matrix_wrapper<T, layout>> {
  public:
    /// The type of the matrix wrapper to convert from/to.
    using matrix_type = plssvm::bindings::python::util::matrix_wrapper<T, layout>;

    /// Specify the Python type name to which a matrix_wrapper should be converted.
    PYBIND11_TYPE_CASTER(matrix_type, _("numpy.ndarray"));

    /**
     * @brief Convert a matrix_wrapper to a Numpy ndarray. Simply calls the custom type caster for a plssvm::matrix.
     * @param[in] matr the PLSSVM matrix to convert to a Numpy ndarray
     * @params[in] rvp *unused*
     * @params[in] h *unused*
     * @return a Pybind11 handle to the Numpy ndarray
     */
    static py::handle cast(const matrix_type &matr, [[maybe_unused]] const py::return_value_policy rvp, [[maybe_unused]] const py::handle h) {
        return py::cast(matr.matrix);
    }

    /**
     * @brief Try converting a Python object @p obj to a matrix_wrapper.
     * @detauls Calls the custom type caster for a plssvm::matrix and, additionally, tries to gather the feature names.
     * @param[in] obj the object to convert
     * @params[in] allow_implicit_conversions *unused*
     * @return `true` if the conversion was successful, `false` otherwise
     * @throws py::value_error all exceptions from the custom plssvm::matrix type caster
     * @throws py::value_error if not all column names are strings
     */
    bool load(py::handle obj, [[maybe_unused]] const bool allow_implicit_conversions) {
        // convert the object to a plssvm::matrix
        value.matrix = obj.cast<plssvm::matrix<T, layout>>();

        if (plssvm::bindings::python::util::is_pandas_data_frame(obj)) {
            // check whether column names can be set
            if (py::hasattr(obj, "columns")) {
                const auto &list = obj.attr("columns").cast<py::list>();
                std::vector<std::string> column_names{};
                column_names.reserve(list.size());
                for (const py::handle &item : list) {
                    // note: column names are only set if they are ALL strings
                    if (!py::isinstance<py::str>(item)) {
                        throw py::type_error{
                            "Feature names are only supported if all input features have string names. If you want feature names to be stored and validated, "
                            "you must convert them all to strings, by using X.columns = X.columns.astype(str) for example. Otherwise you can remove feature / "
                            "column names from your input data, or convert them all to a non-string data type."
                        };
                    }
                    column_names.push_back(item.cast<std::string>());
                }
                // set the column names in the matrix_wrapper
                value.feature_names = std::make_optional(std::move(column_names));
            } else {
                value.feature_names = std::nullopt;
            }
        }

        return true;
    }
};

}  // namespace pybind11::detail

#endif  // PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_WRAPPER_TYPE_CASTER_HPP_
