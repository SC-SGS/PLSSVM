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
#include "plssvm/detail/utility.hpp"         // plssvm::always_false_v

#include "gtest/gtest.h"  // ::testing::Types

#include <fstream>      // std::ifstream, std::ofstream
#include <string>       // std::string
#include <type_traits>  // std::is_same_v, std::is_signed_v, std::is_unsigned_v, std::is_floating_point_v
#include <utility>      // std::pair, std::make_pair

namespace util {

// struct for the used type combinations
template <typename T, typename U>
struct type_combinations {
    using real_type = T;
    using label_type = U;
};

// the floating point and label types combinations to test
using type_combinations_types = ::testing::Types<
    type_combinations<float, bool>,
    type_combinations<float, char>,
    type_combinations<float, signed char>,
    type_combinations<float, unsigned char>,
    type_combinations<float, short>,
    type_combinations<float, unsigned short>,
    type_combinations<float, int>,
    type_combinations<float, unsigned int>,
    type_combinations<float, long>,
    type_combinations<float, unsigned long>,
    type_combinations<float, long long>,
    type_combinations<float, unsigned long long>,
    type_combinations<float, std::string>,
    type_combinations<double, bool>,
    type_combinations<double, char>,
    type_combinations<double, signed char>,
    type_combinations<double, unsigned char>,
    type_combinations<double, short>,
    type_combinations<double, unsigned short>,
    type_combinations<double, int>,
    type_combinations<double, unsigned int>,
    type_combinations<double, long>,
    type_combinations<double, unsigned long>,
    type_combinations<double, long long>,
    type_combinations<double, unsigned long long>,
    type_combinations<double, std::string>>;

/**
 * @brief Get two distinct labels based on the provided label type.
 * @tparam T the label type
 * @return two distinct label (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline std::pair<T, T> get_distinct_label() {
    if constexpr (std::is_same_v<T, bool>) {
        return std::make_pair(true, false);
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
        plssvm::detail::always_false_v<T>;
    }
}

template <typename T>
inline void instantiate_template_file(const std::string &template_filename, const std::string &output_filename) {
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