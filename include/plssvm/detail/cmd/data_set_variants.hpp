/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements utility functions to create a data set based on command line runtime parameters.
 */

#ifndef PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
#define PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
#pragma once

#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/data_set/data_set.hpp"                 // plssvm::data_set
#include "plssvm/data_set/regression_data_set.hpp"      // plssvm::regression_data_set
#include "plssvm/detail/cmd/parser_predict.hpp"         // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"           // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"           // plssvm::detail::cmd::parser_train
#include "plssvm/detail/utility.hpp"                    // plssvm::detail::unreachable
#include "plssvm/svm_types.hpp"                         // plssvm::svm_type

#include <string>   // std::string
#include <variant>  // std::variant

namespace plssvm::detail::cmd {

/**
 * @brief Only a small number of label types are allowed in the command line invocation: for the classification task `int` and `std::string` and for the regression task the current `real_type`.
 */
using data_set_variants = std::variant<plssvm::classification_data_set<int>, plssvm::classification_data_set<std::string>, plssvm::regression_data_set<real_type>>;

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_train command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_train &cmd_parser) {
    switch (cmd_parser.svm) {
        case svm_type::csvc:
            return data_set_variants{ classification_data_set<label_type>{ cmd_parser.input_filename } };
        case svm_type::csvr:
            return data_set_variants{ regression_data_set<label_type>{ cmd_parser.input_filename } };
    }
    // can never be reached
    ::plssvm::detail::unreachable();
}

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_predict command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_predict &cmd_parser) {
    // TODO: data set type
    return data_set_variants{ classification_data_set<label_type>{ cmd_parser.input_filename } };
}

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_scale command line options.
 * @tparam label_type the type of the labels
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename label_type = typename data_set<>::label_type>
[[nodiscard]] inline data_set_variants data_set_factory_impl(const cmd::parser_scale &cmd_parser) {
    // TODO: always use classification_data_set?
    if (!cmd_parser.restore_filename.empty()) {
        typename plssvm::data_set<label_type>::scaling scale{ cmd_parser.restore_filename };
        return data_set_variants{ classification_data_set<label_type>{ cmd_parser.input_filename, scale } };
    } else {
        typename plssvm::data_set<label_type>::scaling scale{ cmd_parser.lower, cmd_parser.upper };
        return data_set_variants{ classification_data_set<label_type>{ cmd_parser.input_filename, scale } };
    }
}

/**
 * @brief Based on the provided command line @p cmd_parser, return the correct plssvm::data_set.
 * @tparam cmd_parser_type the type of the command line parser (train, predict, or scale)
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
template <typename cmd_parser_type>
[[nodiscard]] inline data_set_variants data_set_factory(const cmd_parser_type &cmd_parser) {
    if (cmd_parser.svm == svm_type::csvr) {
        return data_set_factory_impl<real_type>(cmd_parser);
    } else if (cmd_parser.strings_as_labels) {
        return data_set_factory_impl<std::string>(cmd_parser);
    } else {
        return data_set_factory_impl(cmd_parser);
    }
}

}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
