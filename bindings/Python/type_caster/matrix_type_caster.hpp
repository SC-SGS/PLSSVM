/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom type caster for a plssvm::matrix.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_TYPE_CASTER_HPP_
#define PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_TYPE_CASTER_HPP_
#pragma once

#include "plssvm/constants.hpp"                 // plssvm::PADDING_SIZE
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/matrix.hpp"                    // plssvm::matrix, plssvm::layout_type
#include "plssvm/shape.hpp"                     // plssvm::shape

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{is_pandas_data_frame, is_scipy_sparse_matrix, is_c_contiguous, is_f_contiguous}

#include "fmt/format.h"         // fmt::format
#include "pybind11/cast.h"      // pybind11::detail::type_caster
#include "pybind11/gil.h"       // py::gil_scoped_release
#include "pybind11/numpy.h"     // py::array, py::array_t, py::array::c_style, py::array::f_style
#include "pybind11/pybind11.h"  // py::buffer_info, py::isinstance, py::value_error
#include "pybind11/pytypes.h"   // py::list, py::str

#include <cstddef>      // std::size_t
#include <cstring>      // std::memcpy
#include <string>       // std::string
#include <type_traits>  // std::conditional_t

namespace py = pybind11;

namespace pybind11::detail {

/**
 * @brief A custom Pybind11 type caster to convert Python object from and to a plssvm::matrix.
 * @tparam T the value type of the PLSSVM matrix
 * @tparam layout the memory layout type of the PLSSVM matrix
 */
template <typename T, plssvm::layout_type layout>
struct type_caster<plssvm::matrix<T, layout>> {
  public:
    /// The type of the matrix to convert from/to.
    using matrix_type = plssvm::matrix<T, layout>;

    /// Specify the Python type name to which a plssvm::matrix should be converted.
    PYBIND11_TYPE_CASTER(matrix_type, _("numpy.ndarray"));

    /**
     * @brief Convert a plssvm::matrix to a Numpy ndarray.
     * @details If the PLSSVM matrix's memory layout is AoS, uses a Numpy ndarray with c_style layout,
     *          if the PLSSVM matrix's memory layout is SoA, uses a Numpy ndarray with f_style layout.
     * @param[in] matr the PLSSVM matrix to convert to a Numpy ndarray
     * @params[in] rvp *unused*
     * @params[in] h *unused*
     * @return a Pybind11 handle to the Numpy ndarray
     */
    static py::handle cast(const matrix_type &matr, [[maybe_unused]] const py::return_value_policy rvp, [[maybe_unused]] const py::handle h) {
        const std::size_t num_data_points = matr.num_rows();
        const std::size_t num_features = matr.num_cols();

        // determine the numpy array type based on the matrix layout
        using py_array_type = std::conditional_t<layout == plssvm::layout_type::aos, py::array_t<T, py::array::c_style>, py::array_t<T, py::array::f_style>>;

        // create the Python numpy array
        py_array_type arr({ num_data_points, num_features });
        const py::buffer_info buffer = arr.request();
        T *ptr = static_cast<T *>(buffer.ptr);

        // check if the provided matrix has padding entries -> must be removed
        if (matr.is_padded()) {
            // padding entries found -> copy data row-wise to the Python numpy array
            if constexpr (layout == plssvm::layout_type::aos) {
                for (std::size_t row = 0; row < num_data_points; ++row) {
                    std::memcpy(ptr + row * num_features, matr.data() + row * matr.num_cols_padded(), num_features * sizeof(T));
                }
            } else {
                for (std::size_t col = 0; col < num_features; ++col) {
                    std::memcpy(ptr + col * num_data_points, matr.data() + col * matr.num_rows_padded(), num_data_points * sizeof(T));
                }
            }
        } else {
            // can memcpy data directly
            std::memcpy(ptr, matr.data(), matr.size() * sizeof(T));
        }

        // transfer ownership to Python
        return arr.release();
    }

    /**
     * @brief Convert the @p arr to a plssvm::matrix and set the type caster's internal value.
     * @tparam Flags the Pybind11 Numpy array flags used in @p arr
     * @param[in] arr the Numpy array to convert
     * @return `true` if the conversion was successful, `false` otherwise
     */
    template <auto Flags>
    bool copy_pyarray_to_matrix(const py::array_t<T, Flags> &arr) {
        // get dimensions
        const std::size_t num_rows = arr.shape(0);
        const std::size_t num_cols = arr.shape(1);

        // get the underlying raw memory
        const py::buffer_info buffer = arr.request();
        const T *ptr = static_cast<T *>(buffer.ptr);

        // note: the conversions use OpenMP -> remove Python's Global Interpreter Lock
        const py::gil_scoped_release release;

        // check the memory layout of the Python Numpy array
        if constexpr (static_cast<bool>(Flags & py::array::c_style)) {  // NOLINT(hicpp-signed-bitwise): Pybind11 way to do this
            // the provided Python Numpy array has C style layout
            if constexpr (layout == plssvm::layout_type::aos) {
                // memory layout of Python Numpy array and PLSSVM matrix are the same -> can use memcpy to convert
#pragma omp parallel for
                for (std::size_t row = 0; row < num_rows; ++row) {
                    std::memcpy(value.data() + row * value.num_cols_padded(), ptr + row * num_cols, num_cols * sizeof(T));
                }
            } else if constexpr (layout == plssvm::layout_type::soa) {
                // the memory layouts don't match -> must use loops to convert layouts
#pragma omp parallel for collapse(2)
                for (std::size_t row = 0; row < num_rows; ++row) {
                    for (std::size_t col = 0; col < num_cols; ++col) {
                        value(row, col) = ptr[row * num_cols + col];
                    }
                }
            } else {
                // unsupported PLSSVM matrix memory layout
                return false;
            }
        } else if constexpr (static_cast<bool>(Flags & py::array::f_style)) {  // NOLINT(hicpp-signed-bitwise): Pybind11 way to do this
            if constexpr (layout == plssvm::layout_type::aos) {
                // the memory layouts don't match -> must use loops to convert layouts
#pragma omp parallel for collapse(2)
                for (std::size_t row = 0; row < num_rows; ++row) {
                    for (std::size_t col = 0; col < num_cols; ++col) {
                        value(row, col) = ptr[col * num_rows + row];
                    }
                }
            } else if constexpr (layout == plssvm::layout_type::soa) {
                // memory layout of Python Numpy array and PLSSVM matrix are the same -> can use memcpy to convert
#pragma omp parallel for
                for (std::size_t row = 0; row < num_cols; ++row) {
                    std::memcpy(value.data() + row * value.num_rows_padded(), ptr + row * num_rows, num_rows * sizeof(T));
                }
            } else {
                // unsupported PLSSVM matrix memory layout
                return false;
            }
        } else {
            // should not be reached since we fix this case already in the callee function
            return false;
        }

        return true;
    }

    /**
     * @brief Try converting a Python object @p obj to a plssvm::matrix.
     * @detauls Honors different Numpy ndarray memory layouts (c_style or f_style) and PLSSVM matrix layout types.
     * @param[in] obj the object to convert
     * @params[in] allow_implicit_conversion *unused*
     * @return `true` if the conversion was successful, `false` otherwise
     * @throws py::value_error if the provided Python list is empty (or one-dimensional)
     * @throws py::value_error if the provided 2D Python list has inhomogeneous shape
     * @throws py::value_error if @p obj is not a Numpy ndarray, Pandas DataFrame, SciPy sparse matrix, or Python 2D list
     * @throws py::value_error if the Numpy ndarray doesn't have a two-dimensional shape
     */
    bool load(py::handle obj, [[maybe_unused]] const bool allow_implicit_conversion) {
        // special case py::list
        if (py::isinstance<py::list>(obj)) {
            // provided obj is a Python list -> check if it is a correct py::list of py::list
            // convert to py::list
            const auto &list = obj.cast<py::list>();
            if (list.empty()) {
                throw py::value_error{ "Expected 2D array, got 1D array instead!" };
            }

            // iterate over py::list
            const std::size_t num_rows = list.size();
            const std::size_t num_cols = list[0].cast<py::list>().size();

            // create the matrix with the expected size
            value = matrix_type{ plssvm::shape{ num_rows, num_cols }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

            // fill the matrix
            for (std::size_t row = 0; row < num_rows; ++row) {
                // get the sublist
                auto sublist = list[row].cast<py::list>();
                // check if the number of values in the sublist is correct
                if (num_cols != sublist.size()) {
                    throw py::value_error{ "setting an array element with a sequence. The requested array has an inhomogeneous shape." };
                }
                // add list values to the result matrix
                for (std::size_t col = 0; col < num_cols; ++col) {
                    if (py::isinstance<py::str>(sublist[col])) {
                        // cast py::str to a type T via using a std::string
                        value(row, col) = plssvm::detail::convert_to<T>(sublist[col].cast<std::string>());
                    } else {
                        // directly cast the value to a type T
                        value(row, col) = sublist[col].cast<T>();
                    }
                }
            }
        } else {
            py::array arr{};
            if (py::isinstance<py::array>(obj)) {
                // provided obj is a numpy array
                arr = obj.cast<py::array>();
            } else if (plssvm::bindings::python::util::is_pandas_data_frame(obj)) {
                // provided obj is a Pandas DataFrame
                arr = obj.attr("values").cast<py::array>();
            } else if (plssvm::bindings::python::util::is_scipy_sparse_matrix(obj)) {
                // provided obj is a SciPy sparse matrix
                arr = obj.attr("toarray")().cast<py::array>();
            } else {
                throw py::value_error{ fmt::format("Unsupported data type: {}", std::string{ py::str(py::type::of(obj).attr("__name__")) }) };
            }

            // sanity check the number of elements in the numpy array
            if (arr.ndim() > 2) {
                throw py::value_error{ fmt::format("Found array with dim {}. SVC expected <= 2.", arr.ndim()) };
            }
            if (arr.ndim() == 1) {
                throw py::value_error{ "Expected 2D array, got 1D array instead." };
            }
            if (arr.size() == 0) {
                throw py::value_error{ fmt::format("Found array with 0 sample(s) (shape=({}, {})) while a minimum of 1 is required by SVC.", arr.shape(0), arr.shape(1)) };
            }

            // get dimensions
            const std::size_t num_rows = arr.shape(0);
            const std::size_t num_cols = arr.shape(1);

            // create PLSSVM matrix with the correct dimensions AND padding entries
            value = matrix_type{ plssvm::shape{ num_rows, num_cols }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

            // get the underlying buffer
            const py::buffer_info buffer = arr.request();

            // check the memory layout of the Python Numpy array
            if (plssvm::bindings::python::util::is_c_contiguous<T>(buffer)) {
                // array is already c_style -> no need to force cast
                return copy_pyarray_to_matrix(arr.cast<py::array_t<T, py::array::c_style>>());
            }
            if (plssvm::bindings::python::util::is_f_contiguous<T>(buffer)) {
                // array is already f_style -> no need to force cast
                return copy_pyarray_to_matrix(arr.cast<py::array_t<T, py::array::f_style>>());
            }
            // array is non-contiguous
            if constexpr (layout == plssvm::layout_type::aos) {
                // if we want to get a PLSSVM matrix in AoS layout, force casting to c_style is more performant
                return copy_pyarray_to_matrix(arr.cast<py::array_t<T, py::array::c_style | py::array::forcecast>>());
            }
            if constexpr (layout == plssvm::layout_type::soa) {
                // if we want to get a PLSSVM matrix in SoA layout, force casting to f_style is more performant
                return copy_pyarray_to_matrix(arr.cast<py::array_t<T, py::array::f_style | py::array::forcecast>>());
            }
            return false;
        }

        return true;
    }
};

}  // namespace pybind11::detail

#endif  // PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MATRIX_TYPE_CASTER_HPP_
