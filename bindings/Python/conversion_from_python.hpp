/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for converting Python types to the corresponding C++ types.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_CONVERSION_FROM_PYTHON_HPP_
#define PLSSVM_BINDINGS_PYTHON_CONVERSION_FROM_PYTHON_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/matrix.hpp"     // plssvm::matrix, plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/shape.hpp"      // plssvm::shape

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::detail::is_label_type_in_variant_v

#include "pybind11/numpy.h"     // py::array, py::array_t, py::array::c_style, py::array::f_style, py::buffer_info
#include "pybind11/pybind11.h"  // py::list

#include <cstddef>        // std::size_t
#include <cstdint>        // fixed-width integers
#include <optional>       // std::optional, std::nullopt
#include <string>         // std::string
#include <type_traits>    // std::is_same_v, std::false_type
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::pair, std::make_pair, std::move
#include <variant>        // std::variant
#include <vector>         // std::vector

namespace py = pybind11;

namespace plssvm::bindings::python::util {

/**
 * @brief Convert a Python Numpy array to a `std::vector<T>`.
 * @tparam T the type in the array
 * @param[in] vec the Python Numpy array to convert
 * @return the `std::vector<T>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<T> pyarray_t_to_vector(const py::array_t<T, py::array::c_style | py::array::forcecast> &vec) {
    // check dimensions
    if (vec.ndim() != 1) {
        throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", vec.ndim()) };
    }

    if (vec.size() == 0) {
        // return an empty vector
        return std::vector<T>{};
    } else {
        // convert py::array to std::vector
        return std::vector<T>(vec.data(0), vec.data(0) + vec.shape(0));
    }
}

#define PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(data_type)                             \
    if constexpr (detail::is_label_type_in_variant_v<data_type, possible_vector_types>) { \
        if (type.equal(py::dtype::of<data_type>())) {                                     \
            return pyarray_t_to_vector<data_type>(vec);                                   \
        }                                                                                 \
    }

/**
 * @brief Convert a generic Python Numpy array to a `std::vector<T>`.
 * @param[in] vec the generic Python Numpy array to convert
 * @return a `std::variant` containing the converted `std::vector` (`[[nodiscard]]`)
 */
template <typename possible_vector_types>
[[nodiscard]] possible_vector_types pyarray_to_vector(const py::array &vec) {
    // sanity check the passed py::array
    if (!(vec.flags() & py::array::c_style)) {
        throw py::attribute_error{ "The py::array must be C-contiguous" };
    }

    // the type used in the py::array
    py::dtype type = vec.dtype();

    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(bool)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::int8_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::uint8_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::int16_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::uint16_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::int32_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::uint32_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::int64_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(std::uint64_t)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(float)
    PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS(double)

    if constexpr (detail::is_label_type_in_variant_v<std::string, possible_vector_types>) {
        if (type.attr("kind").cast<std::string>() == "U") {
            // convert py::array of strings to a std::vector<std::string>
            if (vec.ndim() != 1) {
                throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", vec.ndim()) };
            }

            if (vec.size() == 0) {
                // return an empty vector
                return std::vector<std::string>{};
            } else {
                std::vector<std::string> result;
                result.reserve(vec.shape(0));
                for (py::handle item : vec) {
                    result.push_back(py::cast<std::string>(item));
                }
                return result;
            }
        }
    }

    // if we are here, no correct type has been found -> throw exception
    throw py::value_error{ fmt::format("Unsupported data type: {}!", type.attr("name").cast<std::string>()) };
}

/**
 * @brief Convert a Python List to a `std::vector<T>`.
 * @tparam T the types in the `std::vector`
 * @param[in] list list the Python List to convert
 * @return the `std::vector<T>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<T> pylist_to_vector(const py::list &list) {
    std::vector<T> vec(py::len(list));
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if constexpr (std::is_same_v<T, std::string>) {
            vec[i] = list[i].cast<py::str>().cast<std::string>();
        } else {
            vec[i] = list[i].cast<T>();
        }
    }
    return vec;
}

namespace impl {

/**
 * @brief A hash struct for creating the hash value of a py::type.
 */
struct py_type_hash {
    std::size_t operator()(const py::type &t) const {
        return py::hash(t);
    }
};

/**
 * @brief A comparison struct to check two py::type for equality.
 */
struct py_type_equal {
    bool operator()(const py::type &lhs, const py::type &rhs) const {
        return lhs.is(rhs);
    }
};

}  // namespace impl

#define PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS(np_data_type, cpp_data_type)                              \
    if constexpr (detail::is_label_type_in_variant_v<cpp_data_type, possible_vector_types>) {             \
        if (highest_type.equal(np.attr(np_data_type))) {                                                  \
            return std::make_pair(pylist_to_vector<cpp_data_type>(list), py::dtype::of<cpp_data_type>()); \
        }                                                                                                 \
    }

/**
 * @brief Convert a Python List to a `std::vector`. The `std::vector<>::value_type` depends on the types provided in the py::list.
 * @param[in] list the Python List to convert
 * @return [a `std::variant` containing the converted `std::vector`, the used py::dtype] (`[[nodiscard]]`)
 */
template <typename possible_vector_types>
[[nodiscard]] inline std::pair<possible_vector_types, py::dtype> pylist_to_vector_2(const py::list &list) {
    static const py::module_ np = py::module_::import("numpy");
    // define a precedence map, i.e., we internally use the type in the py::list with the highest precedence value
    // example: [0, 1.3, np.int8(6)] -> the types are [int, float, int8] -> precedences are [8, 10, 2] -> the highest precedence is 10 -> we use float internally
    static const std::unordered_map<py::type, int, impl::py_type_hash, impl::py_type_equal> precedence_map{
        { py::module_::import("builtins").attr("bool"), 0 },
        { np.attr("uint8"), 1 },
        { np.attr("int8"), 2 },
        { np.attr("uint16"), 3 },
        { np.attr("int16"), 4 },
        { np.attr("uint32"), 5 },
        { np.attr("int32"), 6 },
        { np.attr("uint64"), 7 },
        { np.attr("int64"), 8 },
        { py::module_::import("builtins").attr("int"), 8 },
        { np.attr("float32"), 9 },
        { np.attr("float64"), 10 },
        { py::module_::import("builtins").attr("float"), 10 },
        { py::module_::import("builtins").attr("str"), 11 }
    };

    // get the "super" type used internally as defined by the precedence_map
    py::type highest_type{ py::module_::import("builtins").attr("bool") };
    int highest_precedence{ -1 };
    for (std::size_t i = 0; i < py::len(list); ++i) {
        py::object item = list[i];
        py::type type = py::type::of(item);
        int precedence = precedence_map.at(type);
        if (precedence > highest_precedence) {
            highest_precedence = precedence;
            highest_type = type;
        }
    }

    // convert the py::list to a vector of the previously determined type

    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("int8", std::int8_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("uint8", std::uint8_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("int16", std::int16_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("uint16", std::uint16_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("int32", std::int32_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("uint32", std::uint32_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("int64", std::int64_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("uint64", std::uint64_t)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("float32", float)
    PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS("float64", double)

    if constexpr (detail::is_label_type_in_variant_v<bool, possible_vector_types>) {
        if (highest_type.equal(py::module_::import("builtins").attr("bool"))) {
            return std::make_pair(pylist_to_vector<bool>(list), py::dtype::of<bool>());
        }
    }
    if constexpr (detail::is_label_type_in_variant_v<std::int64_t, possible_vector_types>) {
        if (highest_type.equal(py::module_::import("builtins").attr("int"))) {
            return std::make_pair(pylist_to_vector<std::int64_t>(list), py::dtype::of<std::int64_t>());
        }
    }
    if constexpr (detail::is_label_type_in_variant_v<double, possible_vector_types>) {
        if (highest_type.equal(py::module_::import("builtins").attr("float"))) {
            return std::make_pair(pylist_to_vector<double>(list), py::dtype::of<double>());
        }
    }
    if constexpr (detail::is_label_type_in_variant_v<std::string, possible_vector_types>) {
        if (highest_type.equal(py::module_::import("builtins").attr("str"))) {
            return std::make_pair(pylist_to_vector<std::string>(list), py::dtype("U"));
        }
    }

    // if we are here, no correct type has been found -> throw exception
    throw py::value_error{ fmt::format("Unsupported data type: {}!", highest_type.attr("__name__").cast<std::string>()) };
}

/**
 * @brief Convert a Python Numpy array to a `plssvm::aos_matrix<T>`.
 * @tparam T the type in the array
 * @param[in] mat the 2D Python Numpy matrix to convert
 * @return the `plssvm::aos_matrix` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] plssvm::aos_matrix<T> pyarray_to_matrix(const py::array_t<T, py::array::c_style | py::array::forcecast> &mat) {
    // TODO: if C++20 is available, use templated lambdas to also support f_style arrays (template)
    using size_type = typename plssvm::aos_matrix<T>::size_type;
    // check dimensions
    if (mat.ndim() != 2) {
        throw py::value_error{ fmt::format("the provided matrix must have exactly two dimensions but has {}!", mat.ndim()) };
    }

    // convert py::array to plssvm::matrix<T>
    py::buffer_info buffer = mat.request();
    T *ptr = static_cast<T *>(buffer.ptr);
    return plssvm::aos_matrix<T>{ plssvm::shape{ static_cast<size_type>(mat.shape(0)), static_cast<size_type>(mat.shape(1)) }, ptr, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
}

/**
 * @brief Check if the provided Python object @p obj is a Pandas DataFrame.
 * @param[in] obj the Python object to check
 * @return `true` if @p obj is a Pandas DataFrame, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool is_pandas_data_frame(const py::object &obj) {
    try {
        // try importing the pandas module
        const py::module_ pd = py::module_::import("pandas");
        const py::object pd_data_frame = pd.attr("DataFrame");
        // check the instance
        return py::isinstance(obj, pd_data_frame);
    } catch (const py::error_already_set &) {
        // error loading the pandas library -> obj can't be a DataFrame
        return false;
    }
}

/**
 * @brief Check if the provided Python object @p obj is a Pandas Series.
 * @param[in] obj the Python object to check
 * @return `true` if @p obj is a Pandas Series, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool is_pandas_series(const py::object &obj) {
    try {
        // try importing the pandas module
        const py::module_ pd = py::module_::import("pandas");
        const py::object pd_series = pd.attr("Series");
        // check the instance
        return py::isinstance(obj, pd_series);
    } catch (const py::error_already_set &) {
        // error loading the pandas library -> obj can't be a Series
        return false;
    }
}

/**
 * @brief Check if the provided Python object @p obj is a SciPy sparse matrix.
 * @param[in] obj the Python object to check
 * @return `true` if @p obj is a SciPy sparse matrix, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] inline bool is_scipy_sparse_matrix(const py::object &obj) {
    try {
        // try importing the scipy module
        const py::module_ scipy_sparse = py::module_::import("scipy.sparse");
        const py::object spmatrix_class = scipy_sparse.attr("spmatrix");
        // check the instance
        return py::isinstance(obj, spmatrix_class);
    } catch (const py::error_already_set &) {
        // error loading the scipy library -> obj can't be a sparse matrix
        return false;
    }
}

/**
 * @brief Convert the provided Pybind11 object @p obj to a plssvm::aos_matrix.
 * @details The supported object types are: Numpy ndarrays, Pandas DataFrames, SciPy sparse matrices, and 2D Python lists.
 *          If the object is a Pandas DataFrame and column names are set, returns these column names (can later be queried using the SVC `feature_names_in_` attribute).
 * @param[in] obj the Python object to convert
 * @throws py::value_error if the Numpy ndarray has more than two dimensions
 * @throws py::value_error if the Numpy ndarray has only one dimension
 * @throws py::value_error if one dimension in the Numpy ndarray is zero
 * @throws py::value_error if the Pandas DataFrame is empty
 * @throws py::value_error if the provided Python list is empty
 * @throws py::value_error if the provided 2D Python list has different number of elements per sublist
 * @throws py::value_error if the provided @p obj isn't a Numpy ndarray, Pandas DataFrame, SciPy sparse matrix, or Python list
 * @return { the converted plssvm::aos_matrix; if available, the feature names } (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::pair<plssvm::aos_matrix<plssvm::real_type>, std::optional<std::vector<std::string>>> pyobject_to_matrix(const py::object &obj) {
    if (py::isinstance<py::array>(obj)) {
        // provided obj is a numpy array
        // convert to py::array
        const auto &py_array = py::cast<py::array>(obj);

        // sanity check the number of elements in the numpy array
        if (py_array.ndim() > 2) {
            throw py::value_error{ fmt::format("Found array with dim {}. SVC expected <= 2.", py_array.ndim()) };
        }
        if (py_array.ndim() == 1) {
            throw py::value_error{ "Expected 2D array, got 1D array instead." };
        }
        if (py_array.size() == 0) {
            throw py::value_error{ fmt::format("Found array with 0 sample(s) (shape=({}, {})) while a minimum of 1 is required by SVC.", py_array.shape(0), py_array.shape(1)) };
        }

        const auto &py_array_t = py::cast<py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast>>(py_array);
        return std::make_pair(plssvm::bindings::python::util::pyarray_to_matrix(py_array_t), std::nullopt);
    } else if (is_pandas_data_frame(obj)) {
        // provided obj is a Pandas DataFrame
        // convert to py::array_t
        const auto &py_array_t = obj.attr("values").cast<py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast>>();

        // sanity check the number of elements in the Pandas DataFrame
        if (py_array_t.size() == 0) {
            throw py::value_error{ "at least one array or dtype is required" };
        }

        // convert py::array_t to plssvm::matrix
        auto matr = plssvm::bindings::python::util::pyarray_to_matrix(py_array_t);

        // get the feature names (column names) if possible
        if (py::hasattr(obj, "columns")) {
            return std::make_pair(std::move(matr), plssvm::bindings::python::util::pylist_to_vector<std::string>(obj.attr("columns")));
        } else {
            return std::make_pair(std::move(matr), std::nullopt);
        }
    } else if (is_scipy_sparse_matrix(obj)) {
        // provided obj is a SciPy sparse matrix
        // convert to py::array_t
        const auto &py_array_t = obj.attr("toarray")("C").cast<py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast>>();

        return std::make_pair(plssvm::bindings::python::util::pyarray_to_matrix(py_array_t), std::nullopt);
    } else if (py::isinstance<py::list>(obj)) {
        // provided obj is a Python list -> check if it is a correct py::list of py::list
        // convert to py::list
        const auto &list = py::cast<py::list>(obj);
        if (list.empty()) {
            throw py::value_error{ "Expected 2D array, got 1D array instead!" };
        }

        // iterate over py::list
        const std::size_t num_rows = list.size();
        const std::size_t num_cols = list[0].cast<py::list>().size();

        // create the matrix with the expected size
        plssvm::aos_matrix<plssvm::real_type> matrix{ plssvm::shape{ num_rows, num_cols } };

        // fill the matrix
        for (std::size_t row = 0; row < num_rows; ++row) {
            // get the sublist
            const auto &sublist = list[row].cast<py::list>();
            // check if the number of values in the sublist is correct
            if (num_cols != sublist.size()) {
                throw py::value_error{ "setting an array element with a sequence. The requested array has an inhomogeneous shape." };
            }
            // add list values to the result matrix
            for (std::size_t col = 0; col < num_cols; ++col) {
                if (py::isinstance<py::str>(sublist[col])) {
                    // cast py::str to a plssvm::real_type
                    matrix(row, col) = static_cast<plssvm::real_type>(py::float_(sublist[col]));
                } else {
                    matrix(row, col) = sublist[col].cast<plssvm::real_type>();
                }
            }
        }
        return std::make_pair(std::move(matrix), std::nullopt);
    } else {
        throw py::value_error{ fmt::format("Unsupported data type: {}", std::string{ py::str(obj.get_type().attr("__name__")) }) };
    }
}

/**
 * @brief Convert the provided Pybind11 object @p obj to a std::vector.
 * @details The supported object types are: Numpy ndarrays, Pandas Series, Pandas DataFrames, and Python lists. Also returns the data type used for the labels.
 * @param[in] obj the Python object to convert
 * @throws py::value_error if the provided @p obj isn't a Numpy ndarray, Pandas Series, Pandas DataFrame, or Python list
 * @return { the converted std::vector; the data type of the labels } (`[[nodiscard]]`)
 */
template <typename possible_vector_types>
[[nodiscard]] inline std::pair<possible_vector_types, py::dtype> pyobject_to_vector(const py::object &obj) {
    if (py::isinstance<py::array>(obj)) {
        // provided obj is a numpy array
        // convert to py::array
        auto py_array = py::cast<py::array>(obj);
        return std::make_pair(plssvm::bindings::python::util::pyarray_to_vector<possible_vector_types>(py_array), py_array.dtype());
    } else if (is_pandas_series(obj)) {
        // provided obj is a Pandas Series
        // convert to py::array_t
        const auto &py_array_t = obj.attr("values").cast<py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast>>();
        return std::make_pair(plssvm::bindings::python::util::pyarray_to_vector<possible_vector_types>(py_array_t), py_array_t.dtype());
    } else if (is_pandas_data_frame(obj)) {
        // provided obj is a Pandas Series
        // convert to py::array_t
        auto py_array_t = obj.attr("values").cast<py::array_t<plssvm::real_type, py::array::c_style | py::array::forcecast>>();
        py_array_t = py_array_t.reshape({ py_array_t.size() });
        return std::make_pair(plssvm::bindings::python::util::pyarray_to_vector<possible_vector_types>(py_array_t), py_array_t.dtype());
    } else if (py::isinstance<py::list>(obj)) {
        // provided obj is a Python list
        return plssvm::bindings::python::util::pylist_to_vector_2<possible_vector_types>(py::cast<py::list>(obj));
    } else {
        throw py::attribute_error{ fmt::format("Unsupported data type: {}", std::string{ py::str(obj.get_type().attr("__name__")) }) };
    }
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_CONVERSION_FROM_PYTHON_HPP_
