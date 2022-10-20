/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Implements utility functions to create a data set based on runtime parameters.
*/

#ifndef PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
#define PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
#pragma once

#include "plssvm/data_set.hpp"                   // plssvm::data_set
#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train

#include <string>   // std::string
#include <variant>  // std::variant

namespace plssvm::detail::cmd {

/**
 * @brief Four different type combinations are allowed in the command line invocation: `float` + `int`, `float` + `std::string`, `double` + `int`, and `double` + `std::string`.
 */
using data_set_variants = std::variant<plssvm::data_set<float>, plssvm::data_set<float, std::string>, plssvm::data_set<double>, plssvm::data_set<double, std::string>>;

/**
 * @brief Return the correct data set type based on the `plssvm::detail::cmd::parser_train` command line options.
 * @tparam real_type the type of the data points
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parameters
 * @return the data set type based on the provided command line parameter (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_train &cmd_parser) {
    return data_set_variants{ plssvm::data_set<real_type, label_type>{ cmd_parser.input_filename } };
}
/**
 * @brief Return the correct data set type based on the `plssvm::detail::cmd::parser_predict` command line options.
 * @tparam real_type the type of the data points
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parameters
 * @return the data set type based on the provided command line parameter (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_predict &cmd_parser) {
    return data_set_variants{ plssvm::data_set<real_type, label_type>{ cmd_parser.input_filename } };
}
/**
 * @brief Return the correct data set type based on the `plssvm::detail::cmd::parser_scale` command line options.
 * @tparam real_type the type of the data points
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parameters
 * @return the data set type based on the provided command line parameter (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type = typename data_set<real_type>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_scale &cmd_parser) {
    if (!cmd_parser.restore_filename.empty()) {
        return data_set_variants{ plssvm::data_set<real_type, label_type>{ cmd_parser.input_filename, { cmd_parser.restore_filename } } };
    } else {
        return data_set_variants{ plssvm::data_set<real_type, label_type>{ cmd_parser.input_filename, { static_cast<real_type>(cmd_parser.lower), static_cast<real_type>(cmd_parser.upper) } } };
    }
}

/**
 * @brief Based on the provided command line @p cmd_parser, return the correct data set type.
 * @tparam cmd_parser_type the type of the command line parameter (train, predict, or scale)
 * @param[in] cmd_parser the provided command line parameters
 * @return the data set type based on the provided command line parameter (`[[nodiscard]]`)
 */
template <typename cmd_parser_type>
[[nodiscard]] inline data_set_variants data_set_factory(const cmd_parser_type &cmd_parser) {
    if (cmd_parser.float_as_real_type && cmd_parser.strings_as_labels) {
        return data_set_factory_impl<float, std::string>(cmd_parser);
    } else if (cmd_parser.float_as_real_type && !cmd_parser.strings_as_labels) {
        return data_set_factory_impl<float>(cmd_parser);
    } else if (!cmd_parser.float_as_real_type && cmd_parser.strings_as_labels) {
        return data_set_factory_impl<double, std::string>(cmd_parser);
    } else {
        return data_set_factory_impl<double>(cmd_parser);
    }
}
}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_