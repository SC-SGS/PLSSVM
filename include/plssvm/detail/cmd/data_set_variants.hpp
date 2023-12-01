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
 * @brief Two different type combinations are allowed in the command line invocation: `real_type` + `int` and `real_type` + `std::string`.
 */
using data_set_variants = std::variant<plssvm::data_set<int>, plssvm::data_set<std::string>>;

/**
 * @brief Return the correct data set type based on the plssvm::detail::cmd::parser_train command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set type based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_train &cmd_parser) {
    return data_set_variants{ plssvm::data_set<label_type>{ cmd_parser.input_filename } };
}
/**
 * @brief Return the correct data set type based on the plssvm::detail::cmd::parser_predict command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set type based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_predict &cmd_parser) {
    return data_set_variants{ plssvm::data_set<label_type>{ cmd_parser.input_filename } };
}
/**
 * @brief Return the correct data set type based on the plssvm::detail::cmd::parser_scale command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set type based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_scale &cmd_parser) {
    if (!cmd_parser.restore_filename.empty()) {
        return data_set_variants{ plssvm::data_set<label_type>{ cmd_parser.input_filename, { cmd_parser.restore_filename } } };
    } else {
        return data_set_variants{ plssvm::data_set<label_type>{ cmd_parser.input_filename, { cmd_parser.lower, cmd_parser.upper } } };
    }
}

/**
 * @brief Based on the provided command line @p cmd_parser, return the correct plssvm::data_set type.
 * @tparam cmd_parser_type the type of the command line parser (train, predict, or scale)
 * @param[in] cmd_parser the provided command line parser
 * @return the data set type based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename cmd_parser_type>
[[nodiscard]] inline data_set_variants data_set_factory(const cmd_parser_type &cmd_parser) {
    if (cmd_parser.strings_as_labels) {
        return data_set_factory_impl<std::string>(cmd_parser);
    } else {
        return data_set_factory_impl(cmd_parser);
    }
}

}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_