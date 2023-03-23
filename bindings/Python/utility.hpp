#ifndef PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_UTILITY_HPP_
#pragma once

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "plssvm/parameter.hpp"       // plssvm::parameter

#include "fmt/format.h"         // fmt::format
#include "pybind11/pybind11.h"  // py::kwargs, py::value_error, py::exception
#include "pybind11/stl.h"       // support for STL types

#include <exception>    // std::exception_ptr, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace py = pybind11;

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
 * @def PLSSVM_REGISTER_EXCEPTION
 * @brief Register the PLSSVM exception @p exception_type derived from @p base_exception in the Python module
 *        @p py_module using the Python name @p py_exception_name.
 */
#define PLSSVM_REGISTER_EXCEPTION(exception_type, py_module, py_exception_name, base_exception)                  \
    static py::exception<exception_type> py_exception_name(py_module, #py_exception_name, base_exception.ptr()); \
    py::register_exception_translator([](std::exception_ptr p) {                                                 \
        try {                                                                                                    \
            if (p) {                                                                                             \
                std::rethrow_exception(p);                                                                       \
            }                                                                                                    \
        } catch (const exception_type &e) {                                                                      \
            py_exception_name(e.what_with_loc().c_str());                                                        \
        }                                                                                                        \
    });

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
