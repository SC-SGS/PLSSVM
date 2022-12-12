/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief All type combinations that should be tested for a data set (and the corresponding helper functions) including utility functions.
 */

#ifndef PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#define PLSSVM_TESTS_TYPES_TO_TEST_HPP_
#pragma once

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::replace_all
#include "plssvm/detail/type_traits.hpp"     // plssvm::detail::always_false_v
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // ::testing::Types, FAIL()

#include <filesystem>   // std::filesystem::exists
#include <fstream>      // std::ifstream, std::ofstream
#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::is_same_v, std::is_signed_v, std::is_unsigned_v, std::is_floating_point_v
#include <utility>      // std::pair, std::make_pair

namespace util {

/**
 * @brief Encapsulates a combination of a used `real_type` (`float` or `double`) and `label_type` (an arithmetic type or `std::string`).
 * @tparam T the used `real_type`
 * @tparam U the used `label_type`
 */
template <typename T, typename U>
struct real_type_label_type_combination {
    using real_type = T;
    using label_type = U;
};

namespace detail {

// concatenate the types of two tuples in a new tuple
template <typename S, typename T>
struct concat_tuple_types;
/**
 * @brief Concatenate the types of the two tuples to a new tuple type.
 * @tparam FirstTupleTypes the first tuple
 * @tparam SecondTupleTypes the second tuple
 */
template <typename... FirstTupleTypes, typename... SecondTupleTypes>
struct concat_tuple_types<std::tuple<FirstTupleTypes...>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<FirstTupleTypes..., SecondTupleTypes...>;
};
/**
 * @brief Shorthand for the `typename concat_tuple_types<...>::type` type.
 */
template <typename... T>
using concat_tuple_types_t = typename concat_tuple_types<T...>::type;

// calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding real_type_label_type_combination types.
template <typename S, typename T>
struct cartesian_type_product;
/**
 * @brief Calculate the cartesian product of the types in two tuples and return a new tuple with the corresponding real_type_label_type_combination types.
 * @tparam FirstTupleType the first type in the first tuple (used to iterate all tuple types recursively)
 * @tparam FirstTupleRemainingTypes  the remaining types in the first tuple
 * @tparam SecondTupleTypes all types in the second tuple
 */
template <typename FirstTupleType, typename... FirstTupleRemainingTypes, typename... SecondTupleTypes>
struct cartesian_type_product<std::tuple<FirstTupleType, FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>> {
    // the cartesian product of {FirstTupleType} and {SecondTupleTypes...} is a list of real_type_label_type_combination
    using FirstTupleType_cross_SecondTupleTypes = std::tuple<real_type_label_type_combination<FirstTupleType, SecondTupleTypes>...>;

    // the cartesian product of {FirstTupleRemainingTypes...} and {Ts...} (computed recursively)
    using FirstTupleRemainingTypes_cross_SecondTupleTypes = typename cartesian_type_product<std::tuple<FirstTupleRemainingTypes...>, std::tuple<SecondTupleTypes...>>::type;

    // concatenate both products
    using type = concat_tuple_types_t<FirstTupleType_cross_SecondTupleTypes, FirstTupleRemainingTypes_cross_SecondTupleTypes>;
};
// end the recursion if the first tuple does not have any types left
template <typename... SecondTupleTypes>
struct cartesian_type_product<std::tuple<>, std::tuple<SecondTupleTypes...>> {
    using type = std::tuple<>;  // the cartesian product of {}x{...} is {}
};
/**
 * @brief Shorthand for the `typename cartesian_type_product<...>::type` type.
 */
template <typename... T>
using cartesian_type_product_t = typename cartesian_type_product<T...>::type;

// convert the types in a tuple to GoogleTests ::testing::Type
template <typename Ts>
struct tuple_to_gtest_types;
/**
 * @brief Convert the types in a tuple to GoogleTests ::testing::Types.
 * @details For example: converts `std::tuple<int, long, float>` to `::testing::Types<int, long, float>`.
 * @tparam T the types in the tuple
 */
template <typename... T>
struct tuple_to_gtest_types<std::tuple<T...>> {
    using type = ::testing::Types<T...>;
};
/**
 * @brief Shorthand for the `typename tuple_to_gtest_types<...>::type` type.
 */
template <typename T>
using tuple_to_gtest_types_t = typename tuple_to_gtest_types<T>::type;

}  // namespace detail

/// A type list of all supported real types as `std::tuple`.
using real_type_list = std::tuple<float, double>;
/// A type list of all supported real types usable in google tests.
using real_type_gtest = detail::tuple_to_gtest_types_t<real_type_list>;

/// A type list of all supported label types (currently arithmetic types and `std::string`) as `std::tuple`.
using label_type_list = std::tuple<bool, char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, long long, unsigned long long, float, double, long double, std::string>;
/// A type list of all supported label types (currently arithmetic types and `std::string`) usable in google tests.
using label_type_gtest = detail::tuple_to_gtest_types_t<label_type_list>;

/// The cartesian product of all real types and label types as `std::tuple`.
using real_type_label_type_combination_list = detail::cartesian_type_product_t<real_type_list, label_type_list>;
/// The cartesian product of all real types and label types usable in google test
using real_type_label_type_combination_gtest = detail::tuple_to_gtest_types_t<real_type_label_type_combination_list>;

/**
 * @brief Encapsulates a combination of a used `real_type` (`float` or `double`) and a `plssvm::kernel_function_type`.
 * @tparam T the used `real_type`
 * @tparam kernel the `plssvm::kernel_function_type`
 */
template <typename T, plssvm::kernel_function_type kernel>
struct parameter_definition {
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
};

/// A type list of all supported real_type and kernel_function_type combinations.
using real_type_kernel_function_gtest = ::testing::Types<
    parameter_definition<float, plssvm::kernel_function_type::linear>,
    parameter_definition<float, plssvm::kernel_function_type::polynomial>,
    parameter_definition<float, plssvm::kernel_function_type::rbf>,
    parameter_definition<double, plssvm::kernel_function_type::linear>,
    parameter_definition<double, plssvm::kernel_function_type::polynomial>,
    parameter_definition<double, plssvm::kernel_function_type::rbf>>;

/**
 * @brief Get two distinct labels based on the provided label type.
 * @details The distinct label values must be provided in increasing order (for a defined order in `std::map`).
 * @tparam T the label type
 * @return two distinct label (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::pair<T, T> get_distinct_label() {
    if constexpr (std::is_same_v<T, bool>) {
        return std::make_pair(false, true);
    } else if constexpr (sizeof(T) == sizeof(char)) {
        return std::make_pair('a', 'b');
    } else if constexpr (std::is_signed_v<T>) {
        return std::make_pair(-1, 1);
    } else if constexpr (std::is_unsigned_v<T>) {
        return std::make_pair(1, 2);
    } else if constexpr (std::is_floating_point_v<T>) {
        return std::make_pair(-1.5, 1.5);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::make_pair("cat", "dog");
    } else {
        static_assert(plssvm::detail::always_false_v<T>, "Unknown label type provided!");
    }
}

/**
 * @brief Replace the label placeholders in @p template_filename with the labels based on the template type @p T and
 *        write the instantiated template file to @p output_filename.
 * @tparam T the type of the labels to instantiate the file for
 * @param[in] template_filename the file used as template
 * @param[in] output_filename the file to save the instantiate template to
 */
template <typename T>
inline void instantiate_template_file(const std::string &template_filename, const std::string &output_filename) {
    // check whether the template_file exists
    if (!std::filesystem::exists(template_filename)) {
        FAIL() << fmt::format("The template file {} does not exist!", template_filename);
    }
    // get a label pair based on the current label type
    const auto [first_label, second_label] = util::get_distinct_label<T>();
    // read the data set template and replace the label placeholder with the correct labels
    std::ifstream input{ template_filename };
    std::string str((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    plssvm::detail::replace_all(str, "LABEL_1_PLACEHOLDER", fmt::format("{}", first_label));
    plssvm::detail::replace_all(str, "LABEL_2_PLACEHOLDER", fmt::format("{}", second_label));
    // write the data set with the correct labels to the temporary file
    std::ofstream out{ output_filename };
    out << str;
}

}  // namespace util

#endif  // PLSSVM_TESTS_TYPES_TO_TEST_HPP_