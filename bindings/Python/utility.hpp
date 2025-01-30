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

#include "plssvm/constants.hpp"           // plssvm::real_type
#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"      // plssvm::detail::contains
#include "plssvm/gamma.hpp"               // plssvm::gamma_type
#include "plssvm/parameter.hpp"           // plssvm::parameter

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::kwargs, py::value_error, py::isinstance, py::str, py::module_, py::register_exception_translator, py::set_error
#include "pybind11/pytypes.h"   // py::type

#include <cstdint>      // fixed-width integers
#include <exception>    // std::exception_ptr, std::rethrow_exception
#include <sstream>      // std::istringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::is_same_v, std::false_type
#include <utility>      // std::forward
#include <variant>      // std::variant
#include <vector>       // std::vector

namespace py = pybind11;

namespace plssvm::bindings::python::util {

namespace detail {

/**
 * @brief Base case for a type having a label_type typedef.
 */
template <typename T>
struct get_label_type {
    using type = typename T::label_type;
};

/**
 * @brief Specialization for a std::vector.
 */
template <typename T>
struct get_label_type<std::vector<T>> {
    using type = T;
};

/**
 * @brief Get the label type from @p T. If @p T is a std::vector, uses the std::vector<>::value_type, otherwise directly uses the member label_type typedef.
 * @tparam T the type to get the label type from
 */
template <typename T>
using get_label_type_t = typename get_label_type<typename plssvm::detail::remove_cvref_t<T>>::type;

/**
 * @brief Base false case.
 */
template <typename T, typename Variant>
struct is_label_type_in_variant : std::false_type { };

/**
 * @brief Specialization for a std::variant. Checks for the types using logical or via fold expression.
 */
template <typename T, typename... Args>
struct is_label_type_in_variant<T, std::variant<Args...>> {
    constexpr static bool value = ((std::is_same_v<T, get_label_type_t<Args>>) || ...);
};

/**
 * @brief Check whether @p T is a label type in @p Variant.
 * @tparam T the type to check
 * @tparam Variant the variant type that should contain the label type @p T
 */
template <typename T, typename Variant>
constexpr bool is_label_type_in_variant_v = is_label_type_in_variant<T, Variant>::value;

}  // namespace detail

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

/**
 * @def PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING
 * @brief Map the @p type to its Numpy type name pendant @p numpy_name.
 */
#define PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(type, numpy_name) \
    template <>                                                  \
    [[nodiscard]] constexpr std::string_view python_type_name_mapping<type>() { return numpy_name; }

/**
 * @brief Tries to convert the given type to its Numpy name.
 * @details The definition is marked as **deleted** if `T` isn't a valid mapped type.
 * @tparam T the type to convert to a string
 * @return the name of `T` (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr std::string_view python_type_name_mapping() = delete;

// map all our supported types
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(bool, "bool")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(char, "np.byte")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(signed char, "np.int8")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(unsigned char, "np.uint8")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(short, "np.int16")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(unsigned short, "np.uint16")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(int, "np.int32")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(unsigned int, "np.uint32")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(long, "np.int64")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(unsigned long, "np.uint64")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(long long, "np.int64")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(unsigned long long, "np.uint64")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(float, "np.float32")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(double, "np.float64")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(long double, "np.longdouble")
PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING(std::string, "str")

#undef PLSSVM_CREATE_PYTHON_TYPE_NAME_MAPPING

/**
 * @brief Depending on the Python @p type, construct a new @p Instance of the @p PossibleTypes using the provided parameters @p args.
 * @details Checks all supported Python types. If a theoretically supported type is not present in @p PossibleTypes, the type is skipped with a constexpr if.
 * @tparam Instance the type of the object to create
 * @tparam PossibleTypes all possible types that could be created using a std::variant
 * @tparam Args the type of the constructor parameters forwarded to the constructor of @p Instance.
 * @param[in] type the dynamic Python type used to determine the used label type
 * @param[in] args the parameters forwarded to the @p Instance constructor
 * @return the constructed @p Instance wrapped in a std::variant of type @p PossibleTypes (`[[nodiscard]]`)
 */
template <template <typename> typename Instance, typename PossibleTypes, typename... Args>
[[nodiscard]] PossibleTypes create_instance(const py::type type, Args &&...args) {
    const py::module_ np = py::module_::import("numpy");

    // boolean
    if constexpr (detail::is_label_type_in_variant_v<bool, PossibleTypes>) {
        if (type.equal(py::module_::import("builtins").attr("bool"))) {
            return Instance<bool>{ std::forward<Args>(args)... };
        }
    }
    // std::int8_t, signed char
    if constexpr (detail::is_label_type_in_variant_v<std::int8_t, PossibleTypes>) {
        if (type.equal(np.attr("int8"))) {
            return Instance<std::int8_t>{ std::forward<Args>(args)... };
        }
    }
    // std::uint8_t, unsigned char
    if constexpr (detail::is_label_type_in_variant_v<std::uint8_t, PossibleTypes>) {
        if (type.equal(np.attr("uint8"))) {
            return Instance<std::uint8_t>{ std::forward<Args>(args)... };
        }
    }
    // std::int16_t, short
    if constexpr (detail::is_label_type_in_variant_v<std::int16_t, PossibleTypes>) {
        if (type.equal(np.attr("int16"))) {
            return Instance<std::int16_t>{ std::forward<Args>(args)... };
        }
    }
    // std::uint16_t, unsigned short
    if constexpr (detail::is_label_type_in_variant_v<std::uint16_t, PossibleTypes>) {
        if (type.equal(np.attr("uint16"))) {
            return Instance<std::uint16_t>{ std::forward<Args>(args)... };
        }
    }
    // std::int32_t, int
    if constexpr (detail::is_label_type_in_variant_v<std::int32_t, PossibleTypes>) {
        if (type.equal(np.attr("int32"))) {
            return Instance<std::int32_t>{ std::forward<Args>(args)... };
        }
    }
    // std::uint32_t, unsigned int
    if constexpr (detail::is_label_type_in_variant_v<std::uint32_t, PossibleTypes>) {
        if (type.equal(np.attr("uint32"))) {
            return Instance<std::uint32_t>{ std::forward<Args>(args)... };
        }
    }
    // std::int64_t, long, long long
    if constexpr (detail::is_label_type_in_variant_v<std::int64_t, PossibleTypes>) {
        if (type.equal(np.attr("int64")) || type.equal(py::module_::import("builtins").attr("int"))) {
            return Instance<std::int64_t>{ std::forward<Args>(args)... };
        }
    }
    // std::uint64_t, unsigned long, unsigned long long
    if constexpr (detail::is_label_type_in_variant_v<std::uint64_t, PossibleTypes>) {
        if (type.equal(np.attr("uint64"))) {
            return Instance<std::uint64_t>{ std::forward<Args>(args)... };
        }
    }
    // float
    if constexpr (detail::is_label_type_in_variant_v<float, PossibleTypes>) {
        if (type.equal(np.attr("float32"))) {
            return Instance<float>{ std::forward<Args>(args)... };
        }
    }
    // double
    if constexpr (detail::is_label_type_in_variant_v<double, PossibleTypes>) {
        if (type.equal(np.attr("float64")) || type.equal(py::module_::import("builtins").attr("float"))) {
            return Instance<double>{ std::forward<Args>(args)... };
        }
    }
    // std::string
    if constexpr (detail::is_label_type_in_variant_v<std::string, PossibleTypes>) {
        if (type.equal(py::module_::import("builtins").attr("str"))) {
            return Instance<std::string>{ std::forward<Args>(args)... };
        }
    }

    // if we are here, no type match has been found -> throw an exception
    throw py::value_error{ fmt::format("Unsupported label type: {}!", type.attr("__name__").cast<std::string>()) };
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
