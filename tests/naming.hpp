// TODO

#ifndef PLSSVM_TESTS_NAMING_HPP_
#define PLSSVM_TESTS_NAMING_HPP_
#pragma once

#include "plssvm/detail/arithmetic_type_name.hpp"
#include "plssvm/detail/string_utility.hpp"
#include "plssvm/detail/utility.hpp"

#include "exceptions/utility.hpp"

#include "fmt/core.h"
#include "fmt/ostream.h"
#include "gtest/gtest.h"

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace naming {

namespace detail {

template <typename T>
struct is_map {
    static constexpr bool value = false;
};
template <typename Key, typename Value>
struct is_map<std::map<Key, Value>> {
    static constexpr bool value = true;
};
template <typename T>
constexpr bool is_map_v = is_map<T>::value;

template <typename T>
struct is_unordered_map {
    static constexpr bool value = false;
};
template <typename Key, typename Value>
struct is_unordered_map<std::unordered_map<Key, Value>> {
    static constexpr bool value = true;
};
template <typename T>
constexpr bool is_unordered_map_v = is_unordered_map<T>::value;

template <typename T>
struct is_set {
    static constexpr bool value = false;
};
template <typename Key>
struct is_set<std::set<Key>> {
    static constexpr bool value = true;
};
template <typename T>
constexpr bool is_set_v = is_set<T>::value;

template <typename T>
struct is_unordered_set {
    static constexpr bool value = false;
};
template <typename Key>
struct is_unordered_set<std::unordered_set<Key>> {
    static constexpr bool value = true;
};
template <typename T>
constexpr bool is_unordered_set_v = is_unordered_set<T>::value;

template <typename T>
struct is_vector {
    static constexpr bool value = false;
};
template <typename T>
struct is_vector<std::vector<T>> {
    static constexpr bool value = true;
};
template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;

}  // namespace detail

// utility.cpp
class map_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (detail::is_map_v<T>) {
            return "map";
        } else if constexpr (detail::is_unordered_map_v<T>) {
            return "unordered_map";
        } else {
            plssvm::detail::always_false_v<T>;
        }
    }
};
class set_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (detail::is_set_v<T>) {
            return "set";
        } else if constexpr (detail::is_unordered_set_v<T>) {
            return "unordered_set";
        } else {
            plssvm::detail::always_false_v<T>;
        }
    }
};
class vector_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (detail::is_vector_v<T>) {
            return "vector";
        } else {
            plssvm::detail::always_false_v<T>;
        }
    }
};

// exceptions
class exception_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ util::exception_type_name<T>() };
    }
};

// general
class arithmetic_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ plssvm::detail::arithmetic_type_name<T>() };
    }
};

[[nodiscard]] inline std::string escape_string(std::string_view sv) {  // TODO: use on other occurrences!
    std::string str{ sv };
    plssvm::detail::replace_all(str, "-", "_M_");
    plssvm::detail::replace_all(str, " ", "_W_");
    if (str.empty()) {
        str = "EMPTY";
    }
    return str;
}

// INSTANTIATE_TEST_SUITE_P
template <typename T>
const auto pretty_print_escaped_string = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return escape_string(std::string{ std::get<0>(param_info.param) });
};

// replace
template <typename T>
const auto pretty_print_replace = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}__{}__{}",
                       escape_string(std::get<0>(param_info.param)),
                       escape_string(std::get<1>(param_info.param)),
                       escape_string(std::get<2>(param_info.param)));
};

// parameter_*
/**
 * @brief Pretty print a flag and value combination.
 * @details Replaces all "-" in a flag with "", all "-" in a value with "m" (for minus), and all "." in a value with "p" (for point).
 * @tparam T the parameter type used in the test fixture
 * @param[in] param_info the parameter info used for pretty printing the test case name
 * @return the test case name
 */
template <typename T>
const auto pretty_print_parameter_flag_and_value = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = std::get<0>(param_info.param);
    plssvm::detail::replace_all(flag, "-", "");
    // sanitize values for Google Test names
    std::string value = fmt::format("{}", std::get<1>(param_info.param));
    plssvm::detail::replace_all(value, "-", "m");
    plssvm::detail::replace_all(value, ".", "p");
    return fmt::format("{}__{}", flag, value);
};
/**
 * @brief Pretty print a flag.
 * @details Replaces all "-" in a flag with "".
 * @tparam T the parameter type used in the test fixture
 * @param[in] param_info the parameter info used for pretty printing the test case name
 * @return the test case name
 */
template <typename T>
const auto pretty_print_parameter_flag = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = param_info.param;
    plssvm::detail::replace_all(flag, "-", "");
    return fmt::format("{}", flag.empty() ? "EMPTY_FLAG" : flag);
};

// DataSetFactory
template <typename T>
const auto pretty_print_data_set_factory = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("float_as_real_type_{}__strings_as_labels_{}", std::get<0>(param_info.param), std::get<1>(param_info.param));
};

// Sha256
template <typename T>
const auto pretty_print_sha256 = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return std::string{ param_info.param.second };
};

// Version
template <typename T>
const auto pretty_print_version_info = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    std::string exe{ std::get<0>(param_info.param) };
    plssvm::detail::replace_all(exe, "-", "_");
    return fmt::format("{}__{}", exe, std::get<1>(param_info.param) ? "with_backend_info" : "without_backend_info");
};

// Backend
template <typename T>
const auto pretty_print_unsupported_backend_combination = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}_and_{}", fmt::join(std::get<0>(param_info.param), "__"), fmt::join(std::get<1>(param_info.param), "__"));
};
template <typename T>
const auto pretty_print_supported_backend_combination = [](const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}_and_{}_res_{}", fmt::join(std::get<0>(param_info.param), "__"), fmt::join(std::get<1>(param_info.param), "__"), std::get<2>(param_info.param));
};

}  // namespace naming

#endif  // PLSSVM_TESTS_NAMING_HPP_