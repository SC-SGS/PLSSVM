/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions to improve the GTest test case names.
 */

#ifndef PLSSVM_TESTS_NAMING_HPP_
#define PLSSVM_TESTS_NAMING_HPP_
#pragma once

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::replace_all
#include "plssvm/detail/utility.hpp"               // plssvm::detail::always_false_v

#include "exceptions/utility.hpp"  // util::exception_type_name
#include "types_to_test.hpp"       // util::real_type_label_type_combination

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // directly output types with an operator<< overload using fmt
#include "gtest/gtest.h"  // ::testing::TestParamInfo

#include <map>            // std::map
#include <set>            // std::set
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <tuple>          // std::get
#include <type_traits>    // std::is_same_v, std::true_type, std::false_type
#include <unordered_map>  // std::unordered_map
#include <unordered_set>  // std::unordered_set
#include <vector>         // std::vector
#include <filesystem> // std::filesystem::path

namespace naming {

// different type traits used in the naming functions
namespace detail {

// type_trait to check whether the given type is a std::map
template <typename T>
struct is_map : std::false_type {};
template <typename Key, typename Value>
struct is_map<std::map<Key, Value>> : std::true_type {};
template <typename T>
constexpr bool is_map_v = is_map<T>::value;

// type_trait to check whether the given type is a std::unordered_map
template <typename T>
struct is_unordered_map : std::false_type {};
template <typename Key, typename Value>
struct is_unordered_map<std::unordered_map<Key, Value>> : std::true_type {};
template <typename T>
constexpr bool is_unordered_map_v = is_unordered_map<T>::value;

// type_trait to check whether the given type is a std::set
template <typename T>
struct is_set : std::false_type {};
template <typename Key>
struct is_set<std::set<Key>> : std::true_type {};
template <typename T>
constexpr bool is_set_v = is_set<T>::value;

// type_trait to check whether the given type is a std::unordered_set
template <typename T>
struct is_unordered_set : std::false_type {};
template <typename Key>
struct is_unordered_set<std::unordered_set<Key>> : std::true_type {};
template <typename T>
constexpr bool is_unordered_set_v = is_unordered_set<T>::value;

// type_trait to check whether the given type is a std::vector
template <typename T>
struct is_vector : std::false_type {};
template <typename T>
struct is_vector<std::vector<T>> : std::true_type {};
template <typename T>
constexpr bool is_vector_v = is_vector<T>::value;

// type_trait to check whether the given type is a std::tuple
template <typename T>
struct is_tuple : std::false_type {};
template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};
template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

}  // namespace detail

////////////////////////////////////////////////////////////////////////////////
////                     PRETTY PRINT TYPED_TEST TYPES                      ////
////////////////////////////////////////////////////////////////////////////////
// detail/utility.cpp
class map_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (detail::is_map_v<T>) {
            return "map";
        } else if constexpr (detail::is_unordered_map_v<T>) {
            return "unordered_map";
        } else {
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
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
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
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
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
        }
    }
};

// exceptions/exceptions.cpp
class exception_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ util::exception_type_name<T>() };
    }
};

// detail/io/file_reader.cpp
class open_parameter_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<T, const char *>) {
            return "const_char_ptr";
        } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
            return "std_filesystem_path";
        } else if constexpr (std::is_same_v<T, std::string>) {
            return "std_string";
        } else {
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
        }
    }
};

// general TODO: remove?
class arithmetic_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ plssvm::detail::arithmetic_type_name<T>() };
    }
};
class arithmetic_types_or_string_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<T, std::string>) {
            return "string";
        } else {
            return std::string{ plssvm::detail::arithmetic_type_name<T>() };
        }
    }
};

// types_to_test.hpp
class real_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ plssvm::detail::arithmetic_type_name<T>() };
    }
};
class label_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<T, std::string>) {
            return "string";
        } else {
            return std::string{ plssvm::detail::arithmetic_type_name<T>() };
        }
    }
};
class real_type_label_type_combination_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        //static_assert(std::is_same_v<T, real_type_label_type_combination_to_name>, "T must be of type 'real_type_label_type_combination_to_name'.");
        return fmt::format("{}__x__{}", real_type_to_name::GetName<typename T::real_type>(0), label_type_to_name::GetName<typename T::label_type>(0));
    }
};

////////////////////////////////////////////////////////////////////////////////
////                   PRETTY PRINT PARAMETERIZED TESTS                     ////
////////////////////////////////////////////////////////////////////////////////
namespace detail {

/**
 * @brief Escape some characters of the string such that GTest accepts it as test case name.
 * @details Replaces all "-" with "_M_" (for Minus), all " " with "_W_" (for Whitespace), and "." with "_D_" (for dot).
 *          If the resulting string would be emty, returns a string containing "EMPTY".
 * @param[in] sv the string to escape for GTest
 * @return the escaped test case name (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::string escape_string(std::string_view sv) {
    std::string str{ sv };
    plssvm::detail::replace_all(str, "-", "_M_");
    plssvm::detail::replace_all(str, " ", "_W_");
    plssvm::detail::replace_all(str, ".", "_D_");
    plssvm::detail::replace_all(str, "/", "_");
    if (str.empty()) {
        str = "EMPTY";
    }
    return str;
}

}  // namespace detail

// general
/**
 * @brief Either escape the first string in the parameter info @p param_info if it is an `std::tuple` or directly escapes the value in @p param_info.
 * @details Default escapes the string using the `naming::detail::escape_string()` function.
 * @tparam T the test suite type
 * @param[in] param_info the parameters to scape
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_escaped_string(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    if constexpr (detail::is_tuple_v<decltype(param_info.param)>) {
        return detail::escape_string(fmt::format("{}", std::get<0>(param_info.param)));
    } else {
        return detail::escape_string(fmt::format("{}", param_info.param));
    }
}

// detail/string_utility.cpp -> replace
/**
 * @brief Generate a test case name for the `plssvm::detail::replace_all` test. Escapes all necessary strings.
 * @details `plssvm::detail::replace_all("abc", "a", "c")` translates to a test case name `"replace__abc__what__a__with__c"`.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate and escape
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_replace(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("replace__{}__what__{}__with__{}",
                       detail::escape_string(std::get<0>(param_info.param)),
                       detail::escape_string(std::get<1>(param_info.param)),
                       detail::escape_string(std::get<2>(param_info.param)));
}

// detail/cmd/parameter_*.cpp -> parameter_predict, parameter_scale, parameter_train
/**
 * @brief Generate a test case name using a command line flag and value combination.
 * @details Replaces all "-" in a flag with "" and escapes the value string using `naming::detail::escape_string()`.
 * @tparam T the test suite type
 * @param[in] param_info the parameters to aggregate and escape
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_parameter_flag_and_value(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = std::get<0>(param_info.param);
    plssvm::detail::replace_all(flag, "-", "");
    // sanitize values for Google Test names
    std::string value = detail::escape_string(fmt::format("{}", std::get<1>(param_info.param)));
    return fmt::format("{}__{}", flag, value);
}
/**
 * @brief Generate a test case name using a command line flag.
 * @details Replaces all "-" in a flag with "".
 * @tparam T the test suite type
 * @param[in] param_info the parameters to escape
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_parameter_flag(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = param_info.param;
    plssvm::detail::replace_all(flag, "-", "");
    return fmt::format("{}", flag.empty() ? "EMPTY_FLAG" : flag);
}

// detail/cmd/data_set_variants.cpp -> DataSetFactory
/**
 * @brief Generate a test case name for the data set factory tests using the "float_as_real_type" and "strings_as_labels" combinations.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_data_set_factory(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // the values are bools
    return fmt::format("float_as_real_type_{}__strings_as_labels_{}", std::get<0>(param_info.param), std::get<1>(param_info.param));
}

// detail/sha256.cpp -> Sha256
/**
 * @brief Generate a test case name for the sha256 tests using only the expected correct sha256 encoding.
 * @details The string to encode can't be used since it would be too long or not escapable.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_sha256(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // the first value can't be used (too long or not escapable)
    return std::string{ param_info.param.second };
}

// version/version.cpp -> VersionInfo
/**
 * @brief Generate a test case name for getting the executable version info.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_version_info(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // the executable name
    std::string exe{ std::get<0>(param_info.param) };
    plssvm::detail::replace_all(exe, "-", "_");
    return fmt::format("{}__{}", exe, std::get<1>(param_info.param) ? "with_backend_info" : "without_backend_info");
}

// backend.cpp -> BackendTypeUnsupportedCombination
/**
 * @brief Generate a test case name using the unsupported backend type combinations.
 * @details A possible test case name could be: `"cuda__hip__AND__cpu"`.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_unsupported_backend_combination(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}__AND__{}", fmt::join(std::get<0>(param_info.param), "__"), fmt::join(std::get<1>(param_info.param), "__"));
}
// backend.cpp -> BackendTypeSupportedCombination
/**
 * @brief Generate a test case name using the supported backend type combinations together with the expected result.
 * @details A possible test case name could be: `"openmp__AND__cpu__gpu_nvidia__gpu_amd__gpu_intel__RESULT__cpu"`.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_supported_backend_combination(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}__AND__{}__RESULT__{}", fmt::join(std::get<0>(param_info.param), "__"), fmt::join(std::get<1>(param_info.param), "__"), std::get<2>(param_info.param));
}

// default_value.cpp -> DefaultValueRelational
/**
 * @brief Generate a test case name for the tests for the relational operations in the `plssvm::default_value` class.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_default_value_relational(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return std::string{ std::get<1>(param_info.param) };
}

}  // namespace naming

#endif  // PLSSVM_TESTS_NAMING_HPP_