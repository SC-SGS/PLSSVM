/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions used for creating the Pybind11 Python bindings.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "fmt/format.h"         // fmt::format
#include "pybind11/numpy.h"     // py::array_t, py::buffer_info
#include "pybind11/pybind11.h"  // py::kwargs, py::value_error, py::exception, py::str
#include "pybind11/stl.h"       // support for STL types

#include <exception>    // std::exception_ptr, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace py = pybind11;

/**
 * @brief Convert a `std::vector<T>` to a Python Numpy array.
 * @tparam T the type in the array
 * @param[in] vec the vector to convert
 * @return the Python Numpy array (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] py::array_t<T> vector_to_pyarray(const std::vector<T> &vec) {
    py::array_t<T> py_array(vec.size());
    py::buffer_info buffer = py_array.request();
    T *ptr = static_cast<T *>(buffer.ptr);
    for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i) {
        ptr[i] = vec[i];
    }
    return py_array;
}

/**
 * @brief Convert a `std::vector<std::vector<T>>` to a Python Numpy array.
 * @tparam T the type in the array
 * @param[in] mat the 2D vector to convert
 * @return the Python Numpy array (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] py::array_t<T> matrix_to_pyarray(const std::vector<std::vector<T>> &mat) {
    const typename std::vector<std::vector<T>>::size_type num_data_points = mat.size();
    const typename std::vector<std::vector<T>>::size_type num_features = mat.front().size();

    py::array_t<T> py_array({ num_data_points, num_features });
    py::buffer_info buffer = py_array.request();
    T *ptr = static_cast<T *>(buffer.ptr);
    for (typename std::vector<std::vector<T>>::size_type i = 0; i < num_data_points; ++i) {
        for (typename std::vector<std::vector<T>>::size_type j = 0; j < num_features; ++j) {
            ptr[i * num_features + j] = mat[i][j];
        }
    }
    return py_array;
}

/**
 * @brief Convert a Python Numpy array to a `std::vector<T>`.
 * @tparam T the type in the array
 * @param[in] vec the Python Numpy array to convert
 * @return the `std::vector<T>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<T> pyarray_to_vector(py::array_t<T> vec) {
    // check dimensions
    if (vec.ndim() != 1) {
        throw py::value_error{ fmt::format("the provided array must have exactly one dimension but has {}!", vec.ndim()) };
    }

    // convert py::array to std::vector
    return std::vector<T>(vec.data(0), vec.data(0) + vec.shape(0));
}

/**
 * @brief Convert a Python Numpy array to a `std::vector<std::string>`.
 * @tparam T the type in the array
 * @param[in] vec the Python Numpy array to convert
 * @return the `std::vector<std::string>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<std::string> pyarray_to_string_vector(py::array_t<T> vec) {
    // check dimensions
    if (vec.ndim() != 1) {
        throw py::value_error{ fmt::format("the provided array must have exactly one dimension but has {}!", vec.ndim()) };
    }

    // convert labels to strings
    std::vector<std::string> tmp(vec.shape(0));
    for (std::vector<std::string>::size_type i = 0; i < tmp.size(); ++i) {
        tmp[i] = fmt::format("{}", *vec.data(i));
    }

    return tmp;
}

/**
 * @brief Convert a Python List to a `std::vector<std::string>`.
 * @param[in] list the Python List to convert
 * @return the `std::vector<std::string>` (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::vector<std::string> pylist_to_string_vector(py::list list) {
    // convert a Python list containing strings to a std::vector<std::string>
    std::vector<std::string> tmp(py::len(list));
    for (std::vector<std::string>::size_type i = 0; i < tmp.size(); ++i) {
        tmp[i] = list[i].cast<py::str>().cast<std::string>();
    }

    return tmp;
}

/**
 * @brief Convert a Python Numpy array to a `std::vector<std::vector<T>>`.
 * @tparam T the type in the array
 * @param[in] mat the 2D Python Numpy matrix to convert
 * @return to 2D matrix of `std::vector<std::vector<T>>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<std::vector<T>> pyarray_to_matrix(py::array_t<T> mat) {
    // check dimensions
    if (mat.ndim() != 2) {
        throw py::value_error{ fmt::format("the provided matrix must have exactly two dimensions but has {}!", mat.ndim()) };
    }

    // convert py::array to std::vector<std::vector<T>>
    std::vector<std::vector<T>> tmp(mat.shape(0));
    for (typename std::vector<std::vector<T>>::size_type i = 0; i < tmp.size(); ++i) {
        tmp[i] = std::vector<T>(mat.data(i, 0), mat.data(i, 0) + mat.shape(1));
    }

    return tmp;
}

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
 * @brief Register the PLSSVM @p Exception type as an Python exception with the @p py_exception_name derived from @p BaseException.
 * @tparam Exception the PLSSVM exception to register in Python
 * @tparam BaseException the Python base exception
 * @param[in, out] m the module in which the Python exception is located
 * @param[in] py_exception_name the name of the Python exception
 * @param[in] base_exception the Python exception the new exception should be derived from
 */
template <typename Exception, typename BaseException>
void register_py_exception(py::module_ &m, const std::string &py_exception_name, BaseException &base_exception) {
    static py::exception<Exception> py_exception(m, py_exception_name.c_str(), base_exception.ptr());
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const Exception &e) {
            py_exception(e.what_with_loc().c_str());
        }
    });
}

namespace detail {

/**
 * @def PLSSVM_CREATE_NUMPY_NAME_MAPPING
 * @brief Map the @p type to its Numpy type name pendant @p numpy_name.
 */
#define PLSSVM_CREATE_NUMPY_NAME_MAPPING(type, numpy_name) \
    template <>                                            \
    [[nodiscard]] constexpr inline std::string_view numpy_name_mapping<type>() { return numpy_name; }

/**
 * @brief Tries to convert the given type to its Numpy name.
 * @details The definition is marked as **deleted** if `T` isn't a valid mapped type.
 * @tparam T the type to convert to a string
 * @return the name of `T` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr inline std::string_view numpy_name_mapping() = delete;

PLSSVM_CREATE_NUMPY_NAME_MAPPING(bool, "bool")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(char, "char")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(signed char, "byte")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(unsigned char, "ubyte")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(short, "short")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(unsigned short, "ushort")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(int, "intc")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(unsigned int, "uintc")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(long, "int")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(unsigned long, "uint")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(long long, "longlong")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(unsigned long long, "ulonglong")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(float, "float")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(double, "double")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(long double, "longdouble")
PLSSVM_CREATE_NUMPY_NAME_MAPPING(std::string, "string")

#undef PLSSVM_CREATE_NUMPY_NAME_MAPPING

}  // namespace detail

/**
 * @brief Append the type information to the base @p class_name.
 * @tparam real_type the type of the data points to convert to its Numpy name
 * @tparam label_type the type of the labels to convert to its Numpy name
 * @param class_name the base class name (the type names are appended to it)
 * @return the unique class name
 */
template <typename real_type, typename label_type>
[[nodiscard]] inline std::string assemble_unique_class_name(const std::string_view class_name) {
    return fmt::format("{}_{}_{}", class_name, detail::numpy_name_mapping<real_type>(), detail::numpy_name_mapping<label_type>());
}

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
