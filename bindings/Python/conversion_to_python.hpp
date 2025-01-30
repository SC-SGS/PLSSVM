/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for converting C++ types to their corresponding Python types.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_CONVERSION_TO_PYTHON_HPP_
#define PLSSVM_BINDINGS_PYTHON_CONVERSION_TO_PYTHON_HPP_
#pragma once

#include "pybind11/numpy.h"     // py::array, py::array_t, py::array::c_style, py::array::f_style, py::buffer_info
#include "pybind11/pybind11.h"  // py::list

#include <cstring>      // std::memcpy
#include <string>       // std::string
#include <type_traits>  // std::is_same_v, std::conditional_t
#include <vector>       // std::vector

namespace py = pybind11;

namespace plssvm::bindings::python::util {

/**
 * @brief Convert a `std::vector<T>` to a Python Numpy array.
 * @tparam T the type in the array
 * @param[in] vec the vector to convert
 * @return the Python Numpy array (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] py::array vector_to_pyarray(const std::vector<T> &vec) {
    if constexpr (std::is_same_v<T, std::string>) {
        py::list l{};
        for (const std::string &str : vec) {
            l.append(str);
        }
        return py::array{ l };
    } else {
        py::array_t<T, py::array::c_style> arr(vec.size());
        py::buffer_info buffer = arr.request();
        T *ptr = static_cast<T *>(buffer.ptr);
        if constexpr (std::is_same_v<T, bool>) {
            // can't use memcpy with std::vector<bool>
            for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i) {
                ptr[i] = vec[i];
            }
        } else {
            // use plain memcpy
            std::memcpy(ptr, vec.data(), vec.size() * sizeof(T));
        }
        return arr;
    }
}

/**
 * @brief Convert a `plssvm::matrix<T>` to a Python Numpy array.
 * @tparam T the type in the array
 * @param[in] mat the matrix to convert
 * @return the Python Numpy array (`[[nodiscard]]`)
 */
template <typename T, plssvm::layout_type layout>
[[nodiscard]] py::array matrix_to_pyarray(const plssvm::matrix<T, layout> &matr) {
    using size_type = typename plssvm::matrix<T, layout>::size_type;
    const size_type num_data_points = matr.num_rows();
    const size_type num_features = matr.num_cols();

    using py_array_type = std::conditional_t<layout == plssvm::layout_type::aos, py::array_t<T, py::array::c_style>, py::array_t<T, py::array::f_style>>;

    py_array_type arr({ num_data_points, num_features });
    py::buffer_info buffer = arr.request();
    T *ptr = static_cast<T *>(buffer.ptr);
    if (matr.is_padded()) {
        if constexpr (layout == plssvm::layout_type::aos) {
            for (size_type row = 0; row < num_data_points; ++row) {
                std::memcpy(ptr + row * num_features, matr.data() + row * matr.num_cols_padded(), num_features * sizeof(T));
            }
        } else {
            for (size_type row = 0; row < num_features; ++row) {
                std::memcpy(ptr + row * num_data_points, matr.data() + row * matr.num_rows_padded(), num_data_points * sizeof(T));
            }
        }
    } else {
        // can memcpy data directly
        std::memcpy(ptr, matr.data(), matr.size() * sizeof(T));
    }
    return arr;
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_CONVERSION_TO_PYTHON_HPP_
