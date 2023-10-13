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
#include "plssvm/detail/type_traits.hpp"           // plssvm::detail::{always_false_v, is_map_v, is_unordered_map_v, is_set_v, is_unordered_set_v, is_vector_v}

#include "exceptions/utility.hpp"  // util::exception_type_name

#include "fmt/format.h"     // fmt::format, fmt::join
#include "fmt/ostream.h"  // directly output types with an operator<< overload using fmt
#include "gtest/gtest.h"  // ::testing::TestParamInfo

#include <array>        // std::array
#include <filesystem>   // std::filesystem::path
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::get
#include <type_traits>  // std::is_same_v, std::true_type, std::false_type

// TODO: better

namespace naming {

// different type traits used in the naming functions
namespace detail {

/**
 * @brief Type trait to check whether @p T is a `std::tuple`.
 * @tparam T the type to check
 */
template <typename T>
struct is_tuple : std::false_type {};
/**
 * @copybrief naming::detail::is_tuple
 */
template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type {};
/**
 * @copybrief naming::detail::is_tuple
 */
template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

/**
 * @brief Escape some characters of the string such that GTest accepts it as test case name.
 * @details Replaces some special cases for better readability: "-" with "_M_" (for Minus), all " " with "_W_" (for Whitespace), "." with "_D_" (for dot),
 *          ":" with "_C_" (for colon), "/" with "_", and "@" with "_A_" (for at).
 *          Afterwards, if there are still non alphanumeric or underscore characters present, simply replaces them with "_".
 *          If the resulting string would be empty, returns a string containing "EMPTY".
 * @param[in] sv the string to escape for GTest
 * @return the escaped test case name (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::string escape_string(const std::string_view sv) {
    std::string str{ sv };
    // replace some special cases for better readability
    plssvm::detail::replace_all(str, "-", "_M_");
    plssvm::detail::replace_all(str, " ", "_W_");
    plssvm::detail::replace_all(str, ".", "_D_");
    plssvm::detail::replace_all(str, ":", "_S_");
    plssvm::detail::replace_all(str, "/", "_");
    plssvm::detail::replace_all(str, "@", "_A_");

    // replace all remaining characters with '_' that are not alphanumeric values or underscores
    for (char &c : str) {
        if (!std::isalnum(c) && c != '_') {
            c = '_';
        }
    }

    if (str.empty()) {
        str = "EMPTY";
    }
    return str;
}

}  // namespace detail

//*************************************************************************************************************************************//
//                                                    PRETTY PRINT TYPED_TEST TYPES                                                    //
//*************************************************************************************************************************************//
// detail/utility.cpp
/**
 * @brief A class used to map a std::map or std::unordered_map to a readable name in the GTest test case name.
 */
class map_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (plssvm::detail::is_map_v<T>) {
            return "map";
        } else if constexpr (plssvm::detail::is_unordered_map_v<T>) {
            return "unordered_map";
        } else {
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
        }
    }
};
/**
 * @brief A class used to map a std::set or std::unordered_set to a readable name in the GTest test case name.
 */
class set_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (plssvm::detail::is_set_v<T>) {
            return "set";
        } else if constexpr (plssvm::detail::is_unordered_set_v<T>) {
            return "unordered_set";
        } else {
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
        }
    }
};
/**
 * @brief A class used to map a std::vector to a readable name in the GTest test case name.
 */
class vector_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (plssvm::detail::is_vector_v<T>) {
            return "vector";
        } else {
            static_assert(plssvm::detail::always_false_v<T>, "Invalid type for name mapping provided!");
        }
    }
};

// exceptions/exceptions.cpp
/**
 * @brief A class used to map a plssvm::exception (and derived classes) to a readable name in the GTest test case name.
 */
class exception_types_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return std::string{ util::exception_type_name<T>() };
    }
};

// detail/io/file_reader.cpp
/**
 * @brief A class used to map the types that can be used to open a file to a readable name in the GTest test case name.
 */
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

// types_to_test.hpp
/**
 * @brief A class used to map all real types to a readable name in the GTest test case name.
 */
class real_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        std::string name{ plssvm::detail::arithmetic_type_name<T>() };
        return plssvm::detail::replace_all(name, " ", "_");
    }
};
/**
 * @brief A class used to map all legal label types to a readable name in the GTest test case name.
 */
class label_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        if constexpr (std::is_same_v<T, std::string>) {
            return "string";
        } else {
            return real_type_to_name::GetName<T>(0);
        }
    }
};
/**
 * @brief A class used to map all real and label type-combinations to a readable name in the GTest test case name.
 */
class real_type_label_type_combination_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}__x__{}", real_type_to_name::GetName<typename T::real_type>(0), label_type_to_name::GetName<typename T::label_type>(0));
    }
};

/**
 * @brief A class used to map a parameter definition to a readable name in the GTest test case name.
 */
class parameter_definition_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}__{}", label_type_to_name::GetName<typename T::type>(0), T::value);
    }
};

//*************************************************************************************************************************************//
//                                                   PRETTY PRINT PARAMETERIZED TESTS                                                  //
//*************************************************************************************************************************************//
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

// detail/cmd/parameter_*.cpp -> parser_predict, parser_scale, parser_train
/**
 * @brief Generate a test case name using a command line flag and value combination.
 * @details Replaces all "-" in a flag with "" and escapes the value string using naming::detail::escape_string().
 * @tparam T the test suite type
 * @param[in] param_info the parameters to aggregate and escape
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_parameter_flag_and_value(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    // sanitize flags for Google Test names
    std::string flag = std::get<0>(param_info.param);
    plssvm::detail::replace_all(flag, "-", "");
    return fmt::format("{}__{}", flag, detail::escape_string(fmt::format("{}", std::get<1>(param_info.param))));
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
    return fmt::format("strings_as_labels_{}", std::get<0>(param_info.param));
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

// io/libsvm_model_parsing/utility_functions.cpp -> LIBSVMModelUtilityXvsY
/**
 * @brief Generate a test case name for the LIBSVM model parsing utility function `plssvm::detail::io::x_vs_y_to_idx` tests.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_x_vs_y(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}vs{}__WITH__{}__CLASSES__RESULT_IDX__{}", std::get<0>(param_info.param), std::get<1>(param_info.param), std::get<2>(param_info.param), std::get<3>(param_info.param));
}

// io/libsvm_model_parsing/utility_functions.cpp -> LIBSVMModelUtilityAlphaIdx
/**
 * @brief Generate a test case name for the LIBSVM model parsing utility function `plssvm::detail::io::calculate_alpha_idx` tests.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_calc_alpha_idx(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return fmt::format("{}__AND__{}__WITH__{}__RESULT_IDX__{}", std::get<0>(param_info.param), std::get<1>(param_info.param), std::get<2>(param_info.param), std::get<3>(param_info.param));
}

// kernel_function_types -> KernelFunction
/**
 * @brief Generate a test case name for the LIBSVM model parsing utility function `plssvm::detail::io::calculate_alpha_idx` tests.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string pretty_print_kernel_function(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
    return detail::escape_string(fmt::format("{}__{}", std::get<0>(param_info.param), fmt::join(std::get<1>(param_info.param), "__")));
}

}  // namespace naming

#endif  // PLSSVM_TESTS_NAMING_HPP_