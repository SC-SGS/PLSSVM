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

#include "plssvm/constants.hpp"       // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "plssvm/gamma.hpp"           // plssvm::gamma_type
#include "plssvm/matrix.hpp"          // plssvm::matrix, plssvm::layout_type
#include "plssvm/parameter.hpp"       // plssvm::parameter
#include "plssvm/shape.hpp"           // plssvm::shape

#include "fmt/format.h"            // fmt::format
#include "pybind11/buffer_info.h"  // py::buffer_info
#include "pybind11/cast.h"         // py::cast
#include "pybind11/numpy.h"        // py::array_t
#include "pybind11/pybind11.h"     // py::kwargs, py::value_error, py::exception, py::str, py::set_error
#include "pybind11/pytypes.h"      // py::list
#include "pybind11/stl.h"          // support for STL types

#include <cstddef>        // std::size_t
#include <cstdint>        // fixed-width integers
#include <cstring>        // std::memcpy
#include <exception>      // std::exception_ptr, std::rethrow_exception
#include <sstream>        // std::istringstream
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <tuple>          // std::tuple_element_t, std::tuple_size_v
#include <type_traits>    // std::is_same_v, std::conditional_t
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::integer_sequence, std::make_integer_sequence
#include <variant>        // std::variant
#include <vector>         // std::vector

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
        py::array_t<T, py::array::c_style> py_array(vec.size());
        py::buffer_info buffer = py_array.request();
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
        return py_array;
    }
}

/**
 * @brief Convert a `plssvm::matrix<T>` to a Python Numpy array.
 * @tparam T the type in the array
 * @param[in] mat the matrix to convert
 * @return the Python Numpy array (`[[nodiscard]]`)
 */
template <typename T, plssvm::layout_type layout>
[[nodiscard]] py::array matrix_to_pyarray(const plssvm::matrix<T, layout> &mat) {
    using size_type = typename plssvm::matrix<T, layout>::size_type;
    const size_type num_data_points = mat.num_rows();
    const size_type num_features = mat.num_cols();

    using py_array_type = std::conditional_t<layout == plssvm::layout_type::aos, py::array_t<T, py::array::c_style>, py::array_t<T, py::array::f_style>>;

    py_array_type py_array({ num_data_points, num_features });
    py::buffer_info buffer = py_array.request();
    T *ptr = static_cast<T *>(buffer.ptr);
    if (mat.is_padded()) {
        // must remove padding entries before copying to Python numpy array
        const plssvm::matrix<T, layout> mat_without_padding{ mat, plssvm::shape{ 0, 0 } };
        std::memcpy(ptr, mat_without_padding.data(), mat.size() * sizeof(T));
    } else {
        // can memcpy data directly
        std::memcpy(ptr, mat.data(), mat.size() * sizeof(T));
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
[[nodiscard]] std::vector<T> pyarray_t_to_vector(const py::array_t<T, py::array::c_style | py::array::forcecast> &vec) {
    // check dimensions
    if (vec.ndim() != 1) {
        throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", vec.ndim()) };
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
[[nodiscard]] std::vector<std::string> pyarray_t_to_string_vector(const py::array_t<T, py::array::c_style | py::array::forcecast> &vec) {
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

/// The possible vector types supported when converting a py::array to a std::vector.
using possible_vector_types = std::variant<std::vector<bool>,           // np.bool
                                           std::vector<std::int8_t>,    // np.int8
                                           std::vector<std::uint8_t>,   // np.uint8
                                           std::vector<std::int16_t>,   // np.int16
                                           std::vector<std::uint16_t>,  // np.uint16
                                           std::vector<std::int32_t>,   // np.int32
                                           std::vector<std::uint32_t>,  // np.uint32
                                           std::vector<std::int64_t>,   // np.int64
                                           std::vector<std::uint64_t>,  // np.uint64
                                           std::vector<float>,          // np.float32
                                           std::vector<double>,         // np.float64
                                           std::vector<std::string>>;   // np.string

/**
 * @brief Convert a generic Python Numpy array to a `std::vector<T>`.
 * @param[in] vec the generic Python Numpy array to convert
 * @return a `std::variant` containing the converted `std::vector` (`[[nodiscard]]`)
 */
[[nodiscard]] inline possible_vector_types pyarray_to_vector(const py::array &vec) {
    // sanity check the passed py::array
    if (!(vec.flags() & py::array::c_style)) {
        throw py::attribute_error{ "The py::array must be C-contiguous" };
    }

    // the type used in the py::array
    py::dtype type = vec.dtype();

    if (type.is(py::dtype::of<bool>())) {
        return pyarray_t_to_vector<bool>(vec);
    } else if (type.is(py::dtype::of<std::int8_t>())) {
        return pyarray_t_to_vector<std::int8_t>(vec);
    } else if (type.is(py::dtype::of<std::uint8_t>())) {
        return pyarray_t_to_vector<std::uint8_t>(vec);
    } else if (type.is(py::dtype::of<std::int16_t>())) {
        return pyarray_t_to_vector<std::int16_t>(vec);
    } else if (type.is(py::dtype::of<std::uint16_t>())) {
        return pyarray_t_to_vector<std::uint16_t>(vec);
    } else if (type.is(py::dtype::of<std::int32_t>())) {
        return pyarray_t_to_vector<std::int32_t>(vec);
    } else if (type.is(py::dtype::of<std::uint32_t>())) {
        return pyarray_t_to_vector<std::uint32_t>(vec);
    } else if (type.is(py::dtype::of<std::int64_t>())) {
        return pyarray_t_to_vector<std::int64_t>(vec);
    } else if (type.is(py::dtype::of<std::uint64_t>())) {
        return pyarray_t_to_vector<std::uint64_t>(vec);
    } else if (type.is(py::dtype::of<float>())) {
        return pyarray_t_to_vector<float>(vec);
    } else if (type.is(py::dtype::of<double>())) {
        return pyarray_t_to_vector<double>(vec);
    } else if (type.attr("kind").cast<std::string>() == "U") {
        // convert py::array of strings to a std::vector<std::string>
        if (vec.ndim() != 1) {
            throw py::value_error{ fmt::format("The provided array must have exactly one dimension but has {}!", vec.ndim()) };
        }

        std::vector<std::string> result;
        result.reserve(vec.shape(0));
        for (py::handle item : vec) {
            result.push_back(py::cast<std::string>(item));
        }
        return result;
    } else {
        throw py::value_error{ fmt::format("Unsupported data type: {}!", type.attr("name").cast<std::string>()) };
    }
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

/**
 * @brief Convert a Python List to a `std::vector`. The `std::vector<>::value_type` depends on the types provided in the py::list.
 * @param[in] list the Python List to convert
 * @return [a `std::variant` containing the converted `std::vector`, the used py::dtype] (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::pair<possible_vector_types, py::dtype> pylist_to_vector(const py::list &list) {
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
    if (highest_type.is(py::module_::import("builtins").attr("bool"))) {
        return std::make_pair(pylist_to_vector<bool>(list), py::dtype::of<bool>());
    } else if (highest_type.is(np.attr("int8"))) {
        return std::make_pair(pylist_to_vector<std::int8_t>(list), py::dtype::of<std::int8_t>());
    } else if (highest_type.is(np.attr("uint8"))) {
        return std::make_pair(pylist_to_vector<std::uint8_t>(list), py::dtype::of<std::uint8_t>());
    } else if (highest_type.is(np.attr("int16"))) {
        return std::make_pair(pylist_to_vector<std::int16_t>(list), py::dtype::of<std::int16_t>());
    } else if (highest_type.is(np.attr("uint16"))) {
        return std::make_pair(pylist_to_vector<std::uint16_t>(list), py::dtype::of<std::uint16_t>());
    } else if (highest_type.is(np.attr("int32"))) {
        return std::make_pair(pylist_to_vector<std::int32_t>(list), py::dtype::of<std::int32_t>());
    } else if (highest_type.is(np.attr("uint32"))) {
        return std::make_pair(pylist_to_vector<std::uint32_t>(list), py::dtype::of<std::uint32_t>());
    } else if (highest_type.is(np.attr("int64")) || highest_type.is(py::module_::import("builtins").attr("int"))) {
        return std::make_pair(pylist_to_vector<std::int64_t>(list), py::dtype::of<std::int64_t>());
    } else if (highest_type.is(np.attr("uint64"))) {
        return std::make_pair(pylist_to_vector<std::uint64_t>(list), py::dtype::of<std::uint64_t>());
    } else if (highest_type.is(np.attr("float32"))) {
        return std::make_pair(pylist_to_vector<float>(list), py::dtype::of<float>());
    } else if (highest_type.is(np.attr("float64")) || highest_type.is(py::module_::import("builtins").attr("float"))) {
        return std::make_pair(pylist_to_vector<double>(list), py::dtype::of<double>());
    } else if (highest_type.is(py::module_::import("builtins").attr("str"))) {
        // convert py::array of strings to a std::vector<std::string>
        return std::make_pair(pylist_to_vector<std::string>(list), py::dtype("U"));
    } else {
        throw py::value_error{ fmt::format("Unsupported data type: {}!", highest_type.attr("__name__").cast<std::string>()) };
    }
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
    plssvm::aos_matrix<T> tmp{ plssvm::shape{ static_cast<size_type>(mat.shape(0)), static_cast<size_type>(mat.shape(1)) }, ptr, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    return tmp;
}

template <typename T>
[[nodiscard]] plssvm::soa_matrix<T> pyarray_to_soa_matrix(const py::array_t<T, py::array::c_style | py::array::forcecast> &mat) {
    return plssvm::soa_matrix<T>{ pyarray_to_matrix(mat) };  // TODO: better?
}

/**
 * @brief Check that the Python kwargs @p args only contain keyword arguments with names present in @p valid_named_args.
 * @param[in] args the Python keyword arguments
 * @param[in] valid_named_args the valid keyword arguments
 * @throws pybind11::value_error if an illegal keyword arguments has been provided
 */
inline void check_kwargs_for_correctness(const py::kwargs &args, const std::vector<std::string_view> &valid_named_args) {
    for (const auto &[key, value] : args) {
        if (!plssvm::detail::contains(valid_named_args, key.cast<std::string_view>())) {
            throw py::value_error(fmt::format("got an unexpected keyword argument '{}'", key.cast<std::string_view>()));
        }
    }
}

/**
 * @brief Convert the `gamma` Python kwargs @p args to an `plssvm::gamma_type` object.
 * @note Assumes that @p args contains the keyword argument `gamma`!
 * @param[in] args the Python keyword arguments
 * @return the `plssvm::gamma_type` object filled with the keyword @p args (`[[nodiscard]]`)
 */
[[nodiscard]] inline plssvm::gamma_type convert_gamma_kwarg_to_variant(const py::kwargs &args) {
    if (py::isinstance<py::str>(args["gamma"])) {
        // found a string
        const auto str = args["gamma"].cast<std::string>();
        std::istringstream is{ str };
        plssvm::gamma_type gamma;
        is >> gamma;
        if (is.fail()) {
            throw py::value_error{ fmt::format("When 'gamma' is a string, it should be either 'scale' or 'auto'. Got '{}' instead.", gamma) };
        }
        return gamma;
    } else {
        const auto gamma = args["gamma"].cast<plssvm::real_type>();
        if (gamma <= plssvm::real_type{ 0.0 }) {
            throw py::value_error{ fmt::format("gamma value must be > 0; {} is invalid. Use a positive number or use 'auto' to set gamma to a value of 1 / n_features.", gamma) };
        }
        return gamma;
    }
}

/**
 * @brief Convert the Python kwargs @p args to an `plssvm::parameter` object.
 * @param[in] args the Python keyword arguments
 * @param[in] params the baseline parameter
 * @return the `plssvm::parameter` object filled with the keyword @p args (`[[nodiscard]]`)
 */
[[nodiscard]] inline plssvm::parameter convert_kwargs_to_parameter(const py::kwargs &args, plssvm::parameter params = {}) {
    if (args.contains("kernel_type")) {
        params.kernel_type = args["kernel_type"].cast<decltype(params.kernel_type)>();
    }
    if (args.contains("degree")) {
        params.degree = args["degree"].cast<decltype(params.degree)>();
    }
    if (args.contains("gamma")) {
        params.gamma = convert_gamma_kwarg_to_variant(args);
    }
    if (args.contains("coef0")) {
        params.coef0 = args["coef0"].cast<decltype(params.coef0)>();
    }
    if (args.contains("cost")) {
        params.cost = args["cost"].cast<decltype(params.cost)>();
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
            py::set_error(py_exception, e.what_with_loc().c_str());
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
 * @tparam label_type the type of the labels to convert to its Numpy name
 * @param class_name the base class name (the type names are appended to it)
 * @return the unique class name
 */
template <typename label_type>
[[nodiscard]] inline std::string assemble_unique_class_name(const std::string_view class_name) {
    return fmt::format("{}_{}", class_name, detail::numpy_name_mapping<label_type>());
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @tparam Idx the label type indices
 * @param[in] m the Python module in which the Python bindings are instantiated
 * @param[in] pure_virtual the pure-virtual Python module
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, std::size_t... Idx>
inline void instantiate_module_bindings(py::module_ &m, py::module_ &pure_virtual, std::integer_sequence<std::size_t, Idx...>) {
    (InstantiationFunction<std::tuple_element_t<Idx, LabelTypes>>{}(m, pure_virtual, std::tuple_element_t<Idx, LabelTypes>{}), ...);
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @param[in] m the Python module in which the Python bindings are instantiated
 * @param[in] pure_virtual the pure-virtual Python module
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes>
inline void instantiate_module_bindings(py::module_ &m, py::module_ &pure_virtual) {
    instantiate_module_bindings<InstantiationFunction, LabelTypes>(m, pure_virtual, std::make_integer_sequence<std::size_t, std::tuple_size_v<LabelTypes>>{});
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @tparam Idx the label type indices
 * @param[in] m the Python module in which the Python bindings are instantiated
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, std::size_t... Idx>
inline void instantiate_module_bindings(py::module_ &pure_virtual, std::integer_sequence<std::size_t, Idx...>) {
    (InstantiationFunction<std::tuple_element_t<Idx, LabelTypes>>{}(pure_virtual, std::tuple_element_t<Idx, LabelTypes>{}), ...);
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @param[in] m the Python module in which the Python bindings are instantiated
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes>
inline void instantiate_module_bindings(py::module_ &pure_virtual) {
    instantiate_module_bindings<InstantiationFunction, LabelTypes>(pure_virtual, std::make_integer_sequence<std::size_t, std::tuple_size_v<LabelTypes>>{});
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @tparam PyClassType the type of the Python class to instantiate definitions for
 * @tparam Idx the label type indices
 * @param[in] c the Python class used for instantiation
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, typename PyClassType, std::size_t... Idx>
void instantiate_class_bindings(py::class_<PyClassType> &c, std::integer_sequence<std::size_t, Idx...>) {
    (InstantiationFunction<std::tuple_element_t<Idx, LabelTypes>>{}(c, std::tuple_element_t<Idx, LabelTypes>{}), ...);
}

/**
 * @brief Instantiate Python bindings using the @p InstantiationFunction for all @p LabelTypes.
 * @tparam InstantiationFunction the functor used to instantiate the Python bindings
 * @tparam LabelTypes the label types
 * @tparam PyClassType the type of the Python class to instantiate definitions for
 * @param[in] csvm the Python class used for instantiation
 */
template <template <typename> typename InstantiationFunction, typename LabelTypes, typename PyClassType>
void instantiate_class_bindings(py::class_<PyClassType> &c) {
    instantiate_class_bindings<InstantiationFunction, LabelTypes, PyClassType>(c, std::make_integer_sequence<std::size_t, std::tuple_size_v<LabelTypes>>{});
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
