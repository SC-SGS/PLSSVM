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

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::replace_all
#include "plssvm/detail/type_traits.hpp"           // plssvm::detail::{always_false_v, is_map_v, is_unordered_map_v, is_set_v, is_unordered_set_v, is_vector_v}

#include "exceptions/utility.hpp"  // util::exception_type_name

#include "fmt/format.h"   // fmt::format, fmt::join
#include "fmt/ostream.h"  // directly output types with an operator<< overload using fmt
#include "fmt/ranges.h"   // directly output a std::tuple
#include "gtest/gtest.h"  // ::testing::TestParamInfo

#include <cctype>       // std::isalnum
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::path
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::tuple_element_t, std::tuple_size_v, std::get
#include <type_traits>  // std::true_type, std::false_type, std::is_same_v, std::is_arithmetic_v, std::is_base_of_v

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
 * @brief Type trait used in the trait to check whether a type has a typedef.
 * @tparam T the type
 * @tparam R the type used in the typedef
 */
template <typename T, typename R = void>
struct enable_if_typedef_exists {
    using type = R;
};
/**
 * @brief naming::detail::enable_if_typedef_exists
 */
template <typename T>
using enable_if_typedef_exists_t = typename enable_if_typedef_exists<T>::type;

/**
 * @brief A macro to create type traits for testing whether a type has a typedef called @p def.
 */
#define PLSSVM_CREATE_HAS_MEMBER_TYPEDEF_TYPE_TRAIT(def)                                                   \
    template <typename T, typename Enable = void>                                                          \
    struct has_##def##_member_typedef : std::false_type {};                                                \
    template <typename T>                                                                                  \
    struct has_##def##_member_typedef<T, enable_if_typedef_exists_t<typename T::def>> : std::true_type {}; \
    template <typename T>                                                                                  \
    constexpr bool has_##def##_member_typedef_v = has_##def##_member_typedef<T>::value;

PLSSVM_CREATE_HAS_MEMBER_TYPEDEF_TYPE_TRAIT(csvm_type)
PLSSVM_CREATE_HAS_MEMBER_TYPEDEF_TYPE_TRAIT(device_ptr_type)

#undef PLSSVM_CREATE_HAS_MEMBER_TYPEDEF_TYPE_TRAIT

/**
 * @brief Escape some characters of the string such that GTest accepts it as test case name.
 * @details Replaces some special cases for better readability: "-" with "_M_" (for Minus), " " with "_W_" (for Whitespace), "." with "_D_" (for dot),
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
        if (!std::isalnum(static_cast<int>(c)) && c != '_') {
            c = '_';
        }
    }

    if (str.empty()) {
        str = "EMPTY";
    }
    return str;
}

/**
 * @brief Convert the type @ T to its string representation.
 * @tparam T the type to convert
 * @return the string representation (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string type_name() {
    if constexpr (plssvm::detail::is_map_v<T>) {
        return "std_map";
    } else if constexpr (plssvm::detail::is_unordered_map_v<T>) {
        return "std_unordered_map";
    } else if constexpr (plssvm::detail::is_set_v<T>) {
        return "std_set";
    } else if constexpr (plssvm::detail::is_unordered_set_v<T>) {
        return "std_unordered_set";
    } else if constexpr (plssvm::detail::is_vector_v<T>) {
        return "std_vector";
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "std_string";
    } else if constexpr (std::is_same_v<T, const char *>) {
        return "const_char_ptr";
    } else if constexpr (std::is_same_v<T, std::filesystem::path>) {
        return "std_filesystem_path";
    } else if constexpr (std::is_arithmetic_v<T>) {
        std::string str{ plssvm::detail::arithmetic_type_name<T>() };
        // replace the whitespace in the arithmetic type name with an "_", otherwise it would be replaced by "_W_"
        return plssvm::detail::replace_all(str, " ", "_");
    } else if constexpr (std::is_base_of_v<plssvm::exception, T>) {
        return std::string{ util::exception_type_name<T>() };
    } else if constexpr (has_csvm_type_member_typedef_v<T>) {
        return fmt::format("{}", plssvm::csvm_to_backend_type_v<typename T::csvm_type>);
    } else if constexpr (has_device_ptr_type_member_typedef_v<T>) {
        using device_ptr_type = typename T::device_ptr_type;
        return fmt::format("{}", plssvm::detail::arithmetic_type_name<typename device_ptr_type::value_type>());
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "Can't convert the type 'T' to a std::string!");
    }
}

/**
 * @brief Create the type name string of the type at position @p I in the tuple @p T.
 * @tparam I the position of the current type in the tuple
 * @tparam SIZE the total number of types in the tuple
 * @tparam T the tuple type
 */
template <std::size_t I, std::size_t SIZE, typename T>
struct assemble_tuple_type_string_impl {
    /**
     * @brief Recursively assemble the type name string.
     * @return the string containing all type names separated by "_" (`[[nodiscard]]`)
     */
    [[nodiscard]] static std::string get_name() {
        using type = std::tuple_element_t<I, T>;
        // get the string representation of the currently investigated type
        std::string name{ type_name<type>() };
        // recursively call this function as long as types are present
        if constexpr (I < SIZE - 1) {
            name += "_" + assemble_tuple_type_string_impl<I + 1, SIZE, T>::get_name();
        }
        return name;
    }
};
/**
 * @brief Return a string containing all type names in the provided std::tuple.
 * @details Returns an empty string if no types in the std::tuple are present.
 * @tparam T the std::tuple type
 * @return the type name string (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::string assemble_tuple_type_string() {
    static_assert(is_tuple_v<T>, "The types must be wrapped in a std::tuple!");
    if constexpr (std::tuple_size_v<T> > 0) {
        return assemble_tuple_type_string_impl<0, std::tuple_size_v<T>, T>::get_name();
    } else {
        return "";
    }
}

}  // namespace detail

//*************************************************************************************************************************************//
//                                                    PRETTY PRINT TYPED_TEST TYPES                                                    //
//*************************************************************************************************************************************//

/**
 * @brief Create a test name string from the generic `util::test_parameter` using all its stored types and values.
 */
class test_parameter_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        using used_type_list = typename T::types;
        using used_value_list = typename T::values;

        // assemble the GTest test name string
        const std::string type_names = detail::assemble_tuple_type_string<typename used_type_list::types>();
        const std::string value_names = fmt::format("{}", fmt::join(used_value_list::values, "_"));
        return detail::escape_string(fmt::format("{}{}{}", type_names, !type_names.empty() && !value_names.empty() ? "__" : "", value_names));
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
        return detail::escape_string(fmt::format("{}", fmt::join(param_info.param, "_")));
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
    const auto &[str, what, with, output] = param_info.param;
    return detail::escape_string(fmt::format("replace__{}__what__{}__with__{}__RESULT__", str, what, with, output));
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
    auto [flag, value] = param_info.param;
    return detail::escape_string(fmt::format("{}__{}", plssvm::detail::replace_all(flag, "-", ""), value));
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
    return detail::escape_string(fmt::format("{}", flag.empty() ? "EMPTY_FLAG" : flag));
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
    return detail::escape_string(fmt::format("strings_as_labels_{}", std::get<0>(param_info.param)));
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
    return detail::escape_string(param_info.param.second);
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
    auto [exe, with_backend_info] = param_info.param;
    return detail::escape_string(fmt::format("{}__{}", plssvm::detail::replace_all(exe, "-", "_"), with_backend_info ? "with_backend_info" : "without_backend_info"));
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
    const auto &[backends, target_platforms] = param_info.param;
    return detail::escape_string(fmt::format("{}__AND__{}", fmt::join(backends, "__"), fmt::join(target_platforms, "__")));
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
    const auto &[backends, target_platforms, result] = param_info.param;
    return detail::escape_string(fmt::format("{}__AND__{}__RESULT__{}", fmt::join(backends, "__"), fmt::join(target_platforms, "__"), result));
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
    // the name of the relational operator
    return detail::escape_string(std::get<1>(param_info.param));
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
    const auto &[x, y, num_classes, idx] = param_info.param;
    return detail::escape_string(fmt::format("{}vs{}__WITH__{}__CLASSES__RESULT_IDX__{}", x, y, num_classes, idx));
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
    const auto &[i, j, idx_to_find, expected_global_idx] = param_info.param;
    return detail::escape_string(fmt::format("{}__AND__{}__WITH__{}__RESULT_IDX__{}", i, j, idx_to_find, expected_global_idx));
}

// kernel_function_types -> KernelFunction
/**
 * @brief Generate a test case name for the LIBSVM model parsing utility function `plssvm::detail::io::calculate_alpha_idx` tests.
 * @tparam T the test suite type
 * @param param_info the parameters to aggregate
 * @return the test case name (`[[nodiscard]]`)
 */
// template <typename T>
//[[nodiscard]] inline std::string pretty_print_kernel_function(const ::testing::TestParamInfo<typename T::ParamType> &param_info) {
//     return detail::escape_string(fmt::format("{}__{}", std::get<0>(param_info.param), fmt::join(std::get<1>(param_info.param), "__")));
// }

}  // namespace naming

#endif  // PLSSVM_TESTS_NAMING_HPP_