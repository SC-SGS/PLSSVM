/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implementation of the gamma parameter.
 */

#ifndef PLSSVM_GAMMA_HPP_
#define PLSSVM_GAMMA_HPP_
#pragma once

#include "plssvm/constants.hpp"         // plssvm::real_type
#include "plssvm/detail/operators.hpp"  // plssvm::operators namespace
#include "plssvm/matrix.hpp"            // plssvm::matrix, plssvm::layout_type, plssvm::variance

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>   // forward declare std::ostream and std::istream
#include <string>   // std::string
#include <variant>  // std::variant, std::visit

namespace plssvm {

namespace detail {

/**
 * @brief Struct to overload the `operator()` for multiple std::variant members.
 * @details See: https://en.cppreference.com/w/cpp/utility/variant/visit.
 * @tparam Ts the overloaded types
 */
template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

/**
 * @brief Custom deduction guide for the `overloaded` struct.
 */
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace detail

/**
 * @brief Enum class for all possible gamma coefficient types.
 */
enum class gamma_coefficient_type {
    /** Gamma is dynamically set to 1 / num_features based on the used data set. */
    automatic,
    /** Gamma is dynamically set to 1 / (num_features * data.variance()) based on the used data set. */
    scale
};

/**
 * @brief The type of the gamma value. Either a specific floating point value or a dynamic value based on the used data set distinguished using the `gamma_coefficient_type`.
 */
using gamma_type = std::variant<real_type, gamma_coefficient_type>;

/**
 * @brief Return the correct value of gamma based on the current active variance member.
 * @details Can't use plssvm::data_set directly due to circular dependencies.
 * @tparam T the value type of the matrix
 * @tparam layout the memory layout of the matrix
 * @param[in] var the std::variant holding the type of the gamma to be used
 * @param[in] matr the data used for the `gamma_coefficient_type` gamma values
 * @return the gamma value (`[[nodiscard]]`)
 */
template <typename T, layout_type layout>
[[nodiscard]] real_type calculate_gamma_value(const gamma_type &var, [[maybe_unused]] const matrix<T, layout> &matr) {
    return std::visit(detail::overloaded{
                          [](const real_type val) { return val; },
                          [&](const gamma_coefficient_type val) {
                              switch (val) {
                                  case gamma_coefficient_type::automatic:
                                      return real_type{ 1.0 } / static_cast<real_type>(matr.num_cols());
                                  case gamma_coefficient_type::scale:
                                      return real_type{ 1.0 } / (static_cast<real_type>(matr.num_cols()) * variance(matr));
                              }
                          } },
                      var);
}

/**
 * @brief Return the correct value of gamma based on the current active variance member.
 * @tparam T the value type of the matrix
 * @param[in] var the std::variant holding the type of the gamma to be used
 * @param[in] vec the std::vector values used for the `gamma_coefficient_type` gamma values
 * @return the gamma value (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] real_type calculate_gamma_value(const gamma_type &var, [[maybe_unused]] const std::vector<T> &vec) {
    return std::visit(detail::overloaded{
                          [](const real_type val) { return val; },
                          [&](const gamma_coefficient_type val) {
                              switch (val) {
                                  case gamma_coefficient_type::automatic:
                                      return real_type{ 1.0 } / static_cast<real_type>(vec.size());
                                  case gamma_coefficient_type::scale:
                                      using namespace plssvm::operators;
                                      return real_type{ 1.0 } / (static_cast<real_type>(vec.size()) * variance(vec));
                              }
                          } },
                      var);
}

/**
 * @brief Return the string representing the value of gamma based on the current active variance member.
 * @param[in] var the std::variant holding the type of the gamma to be used
 * @return the string representing the gamma value (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_gamma_string(const gamma_type &var);

/**
 * @brief Output the @p gamma_coefficient to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the gamma_coefficient type to
 * @param[in] gamma_coefficient the gamma coefficient type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, gamma_coefficient_type gamma_coefficient);

/**
 * @brief Use the input-stream @p in to initialize the @p gamma_coefficient type.
 * @param[in,out] in input-stream to extract the gamma_coefficient type from
 * @param[in] gamma_coefficient the gamma coefficient type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, gamma_coefficient_type &gamma_coefficient);

/**
 * @brief Output the @p gamma to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the gamma_value type to
 * @param[in] gamma_value the gamma type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, gamma_type gamma_value);

/**
 * @brief Use the input-stream @p in to initialize the @p gamma_value type.
 * @param[in,out] in input-stream to extract the gamma_value type from
 * @param[in] gamma_value the gamma type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, gamma_type &gamma_value);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::gamma_coefficient_type> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::gamma_type> : fmt::ostream_formatter { };

#endif  // PLSSVM_GAMMA_HPP_
