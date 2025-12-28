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
#include "plssvm/data_set/min_max_scaler.hpp"           // plssvm::min_max_scaler
#include "plssvm/data_set/regression_data_set.hpp"      // plssvm::regression_data_set
#include "plssvm/detail/cmd/parser_predict.hpp"         // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"           // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"           // plssvm::detail::cmd::parser_train
#include "plssvm/detail/utility.hpp"                    // plssvm::detail::unreachable
#include "plssvm/mpi/communicator.hpp"                  // plssvm::mpi::communicator
#include "plssvm/svm_types.hpp"                         // plssvm::svm_type, plssvm::svm_type_from_model_file

#include <string>   // std::string
#include <utility>  // std::move
#include <variant>  // std::variant

namespace plssvm::detail::cmd {

/**
 * @brief Only a small number of label types are allowed in the command line invocation: for the classification task `int` and `std::string` and for the regression task the current `real_type`.
 */
using data_set_variants = std::variant<plssvm::classification_data_set<int>, plssvm::classification_data_set<std::string>, plssvm::regression_data_set<real_type>>;

/**
 * @brief Create a classification data set based on the provided command line arguments.
 * @tparam label_type the type of the labels
 * @tparam Args the types of the arguments
 * @param[in] args the arguments to forward to the classification data set constructor
 * @return the constructed classification data set (`[[nodiscard]]`)
 */
template <typename label_type, typename... Args>
[[nodiscard]] inline data_set_variants make_classification_data_set(Args &&...args) {
    return data_set_variants{ classification_data_set<label_type>{ std::forward<Args>(args)... } };
}

/**
 * @brief Create a regression data set based on the provided command line arguments.
 * @tparam label_type the type of the labels
 * @tparam Args the types of the arguments
 * @param[in] args the arguments to forward to the regression data set constructor
 * @return the constructed regression data set (`[[nodiscard]]`)
 */
template <typename label_type, typename... Args>
[[nodiscard]] inline data_set_variants make_regression_data_set(Args &&...args) {
    return data_set_variants{ regression_data_set<label_type>{ std::forward<Args>(args)... } };
}

/**
 * @brief Create a plssvm::min_max_scaler based on the provided command line arguments.
 * @param[in] comm the MPI communicator wrapper
 * @param[in] cmd_parser the command line arguments
 * @return the constructed plssvm::min_max_scaler (`[[nodiscard]]`)
 */
[[nodiscard]] inline min_max_scaler make_scaling_factors(mpi::communicator comm, const cmd::parser_scale &cmd_parser) {
    if (!cmd_parser.restore_filename.empty()) {
        return min_max_scaler{ std::move(comm), cmd_parser.restore_filename };
    }
    return min_max_scaler{ std::move(comm), cmd_parser.lower, cmd_parser.upper };
}

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_train command line options.
 * @param[in] comm the MPI communicator wrapper
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
[[nodiscard]] inline data_set_variants data_set_factory(mpi::communicator comm, const cmd::parser_train &cmd_parser) {
    switch (cmd_parser.svm) {
        case svm_type::csvc:
            if (cmd_parser.strings_as_labels) {
                return make_classification_data_set<std::string>(std::move(comm), cmd_parser.input_filename);
            }
            return make_classification_data_set<typename classification_data_set<>::label_type>(std::move(comm), cmd_parser.input_filename);
        case svm_type::csvr:
            return make_regression_data_set<typename regression_data_set<>::label_type>(std::move(comm), cmd_parser.input_filename);
    }
    // can never be reached
    ::plssvm::detail::unreachable();
}

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_predict command line options.
 * @details Infers the C-SVM type from the provided model file header.
 * @param[in] comm the MPI communicator wrapper
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
[[nodiscard]] inline data_set_variants data_set_factory(mpi::communicator comm, const cmd::parser_predict &cmd_parser) {
    switch (svm_type_from_model_file(cmd_parser.model_filename)) {
        case svm_type::csvc:
            if (cmd_parser.strings_as_labels) {
                return make_classification_data_set<std::string>(std::move(comm), cmd_parser.input_filename);
            }
            return make_classification_data_set<typename classification_data_set<>::label_type>(std::move(comm), cmd_parser.input_filename);
        case svm_type::csvr:
            return make_regression_data_set<typename regression_data_set<>::label_type>(std::move(comm), cmd_parser.input_filename);
    }
    // can never be reached
    ::plssvm::detail::unreachable();
}

/**
 * @brief Return the correct data set based on the plssvm::detail::cmd::parser_scale command line options.
 * @details **Always** uses a classification data set since it allows more different label types.
 * @param[in] comm the MPI communicator wrapper
 * @param[in] cmd_parser the provided command line parser
 * @return the data set based on the provided command line parser (`[[nodiscard]]`)
 */
[[nodiscard]] inline data_set_variants data_set_factory(mpi::communicator comm, const cmd::parser_scale &cmd_parser) {
    // create the scaling factors
    min_max_scaler scaling_factors = make_scaling_factors(comm, cmd_parser);
    // create the correct classification data set
    if (cmd_parser.strings_as_labels) {
        return make_classification_data_set<std::string>(std::move(comm), cmd_parser.input_filename, std::move(scaling_factors));
    }
    using label_type = typename classification_data_set<>::label_type;
    return make_classification_data_set<label_type>(std::move(comm), cmd_parser.input_filename, std::move(scaling_factors));
}

}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_DATA_SET_VARIANTS_HPP_
