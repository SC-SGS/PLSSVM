/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom type caster for a label_vector_wrapper (storing a std::vector and py::type).
 */

#ifndef PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_LABEL_VECTOR_WRAPPER_TYPE_CASTER_HPP_
#define PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_LABEL_VECTOR_WRAPPER_TYPE_CASTER_HPP_
#pragma once

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::{vector_to_pyarray, is_pandas_data_frame, is_pandas_series}

#include "fmt/format.h"         // fmt::format
#include "pybind11/cast.h"      // pybind11::detail::type_caster
#include "pybind11/numpy.h"     // py::array, py::array_t, py::array::c_style, py::array::f_style, py::buffer_info
#include "pybind11/pybind11.h"  // py::isinstance, py::value_error, py::list, py::hash, py::handle, py::cast, py::len, py::str, py::module_, py::object
#include "pybind11/pytypes.h"   // py::dtype

#include <cstddef>        // std::size_t
#include <cstdint>        // fixed-width integers
#include <string>         // std::string
#include <type_traits>    // std::is_same_v
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::pair, std::make_pair, std::move
#include <variant>        // std::variant, std::visit
#include <vector>         // std::vector

namespace py = pybind11;

namespace plssvm::bindings::python::util {

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

/**
 * @brief Convert a Python Numpy array to a `std::vector<T>`.
 * @tparam T the type in the array
 * @param[in] vec the Python Numpy array to convert
 * @return the `std::vector<T>` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] std::vector<T> pyarray_to_vector(const py::array &arr) {
    // check dimensions
    if (arr.ndim() != 1) {
        throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", arr.ndim()) };
    }

    if (arr.size() == 0) {
        // return an empty vector
        return std::vector<T>{};
    } else {
        // convert py::array to std::vector
        auto arr_t = arr.cast<py::array_t<T>>();
        return std::vector<T>(arr_t.data(0), arr_t.data(0) + arr_t.shape(0));
    }
}

/**
 * @def PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS
 * @brief Create a mapping from a Python Numpy ndarray with the @p data_type to a std::vector.
 */
#define PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(data_type)                               \
    if constexpr (detail::is_label_type_in_variant_v<data_type, possible_vector_types>) { \
        if (type.equal(py::dtype::of<data_type>())) {                                     \
            return pyarray_to_vector<data_type>(arr);                                     \
        }                                                                                 \
    }

/**
 * @brief Convert a generic Python Numpy array to a `std::vector<T>`.
 * @param[in] vec the generic Python Numpy array to convert
 * @return a `std::variant` containing the converted `std::vector` (`[[nodiscard]]`)
 */
template <typename possible_vector_types>
[[nodiscard]] possible_vector_types generic_pyarray_to_vector(const py::array &arr) {
    // sanity check the passed py::array
    if (!(arr.flags() & py::array::c_style)) {
        throw py::value_error{ "The py::array must be C-contiguous" };
    }

    // the type used in the py::array
    py::dtype type = arr.dtype();

    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(bool)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::int8_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::uint8_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::int16_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::uint16_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::int32_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::uint32_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::int64_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(std::uint64_t)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(float)
    PLSSVM_CREATE_PYARRAY_TO_VECTOR_MAPPINGS(double)

    if constexpr (detail::is_label_type_in_variant_v<std::string, possible_vector_types>) {
        if (type.attr("kind").cast<std::string>() == "U") {
            // convert py::array of strings to a std::vector<std::string>
            if (arr.ndim() != 1) {
                throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", arr.ndim()) };
            }

            if (arr.size() == 0) {
                // return an empty vector
                return std::vector<std::string>{};
            } else {
                std::vector<std::string> result;
                result.reserve(arr.shape(0));
                for (py::handle item : arr) {
                    result.push_back(py::cast<std::string>(item));
                }
                return result;
            }
        }
    }

    // if we are here, no correct type has been found -> throw exception
    throw py::value_error{ fmt::format("Unsupported data type: {}!", type.attr("name").cast<std::string>()) };
}

#undef PLSSVM_CREATE_PYARRAY_T_TO_VECTOR_MAPPINGS

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

/**
 * @def PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS
 * @brief Create a mapping from a Python list with the @p np_data_type to a std::vector of @p cpp_data_type.
 */
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
[[nodiscard]] inline std::pair<possible_vector_types, py::dtype> generic_pylist_to_vector(const py::list &list) {
    const py::module_ np = py::module_::import("numpy");
    // define a precedence map, i.e., we internally use the type in the py::list with the highest precedence value
    // example: [0, 1.3, np.int8(6)] -> the types are [int, float, int8] -> precedences are [8, 10, 2] -> the highest precedence is 10 -> we use float internally
    const std::unordered_map<py::type, int, impl::py_type_hash, impl::py_type_equal> precedence_map{
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

#undef PLSSVM_CREATE_PYLIST_TO_VECTOR_MAPPINGS

/**
 * @brief A small wrapper around a std::variant containing all possible label type vectors and the actually used Python dtype.
 * @tparam PossibleTypes the possible label vectors stored in a std::variant
 */
template <typename PossibleTypes>
struct label_vector_wrapper {
    /// The labels.
    PossibleTypes labels{};
    /// The actually used Python dtype.
    py::dtype dtype{};
};

}  // namespace plssvm::bindings::python::util

namespace pybind11::detail {

/**
 * @brief A custom Pybind11 type caster to convert Python object from and to a plssvm::bindings::python::util::label_vector_wrapper.
 * @tparam T the value type of the PLSSVM matrix
 * @tparam layout the memory layout type of the PLSSVM matrix
 */
template <typename PossibleTypes>
struct type_caster<plssvm::bindings::python::util::label_vector_wrapper<PossibleTypes>> {
  public:
    /// The type of the label vector wrapper to convert from/to.
    using label_vector_wrapper_type = plssvm::bindings::python::util::label_vector_wrapper<PossibleTypes>;

    /// Specify the Python type name to which a label_vector_wrapper should be converted.
    PYBIND11_TYPE_CASTER(label_vector_wrapper_type, _("numpy.ndarray"));

    /**
     * @brief Convert a label_vector_wrapper to a Numpy ndarray.
     * @param[in] labels the labels vector to convert to a Python Numpy ndarray
     * @return a Pybind11 handle to the Numpy ndarray
     */
    static handle cast(const label_vector_wrapper_type &labels, return_value_policy, handle) {
        // convert a generic std::vector to a Numpy ndarray
        return std::visit([](auto &&vec) { return plssvm::bindings::python::util::vector_to_pyarray(vec).release(); }, labels.labels);
    }

    /**
     * @brief Try converting a Python object @p obj to a label_vector_wrapper.
     * @param[in] obj the object to convert
     * @return `true` if the conversion was successful, `false` otherwise
     */
    bool load(handle obj, bool) {
        if (py::isinstance<py::list>(obj)) {
            // provided obj is a Python list
            auto [labels, dtype] = plssvm::bindings::python::util::generic_pylist_to_vector<PossibleTypes>(py::cast<py::list>(obj));
            value.labels = std::move(labels);
            value.dtype = dtype;
        } else {
            py::array arr{};
            if (py::isinstance<py::array>(obj)) {
                // provided obj is a numpy array
                arr = obj.cast<py::array>();
            } else if (plssvm::bindings::python::util::is_pandas_series(obj)) {
                // provided obj is a Pandas Series
                arr = obj.attr("values").cast<py::array>();
            } else if (plssvm::bindings::python::util::is_pandas_data_frame(obj)) {
                // provided obj is a Pandas DataFrame
                arr = obj.attr("values").cast<py::array>();
                arr = arr.reshape({ arr.size() });
            } else {
                throw py::value_error{ fmt::format("Unsupported data type: {}", std::string{ py::str(obj.get_type().attr("__name__")) }) };
            }

            // sanity check the number of elements in the numpy array
            if (arr.ndim() != 1) {
                throw py::value_error{ fmt::format("Found array with dim {}. SVC expected == 1.", arr.ndim()) };
            }

            // convert te Python Numpy array to a std::vector
            value.labels = plssvm::bindings::python::util::generic_pyarray_to_vector<PossibleTypes>(arr);
            value.dtype = arr.dtype();
        }

        return true;
    }
};

}  // namespace pybind11::detail

#endif  // PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_LABEL_VECTOR_WRAPPER_TYPE_CASTER_HPP_
