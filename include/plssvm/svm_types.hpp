/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all implemented SVM types.
 */

#ifndef PLSSVM_SVM_TYPES_HPP_
#define PLSSVM_SVM_TYPES_HPP_
#pragma once

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>       // forward declare std::ostream and std::istream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Enum class for all implemented SVM types.
 */
enum class svm_type {
    /** Use a C-SVM for classification. */
    csvc,
    /** Use a C-CVM for regression. */
    csvr
};

/**
 * @brief Return a list of all currently implemented SVM types.
 * @return a list of the implemented SVM types (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<svm_type> list_available_svm_types();

/**
 * @brief Output the @p svm to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the SVM type to
 * @param[in] svm the SVM type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, svm_type svm);

/**
 * @brief Return a task name (e.g., `"classification"` or `"regression"`) based on the provided @p svm.
 * @param[in] svm the type of the C-SVM to retrieve the task name from
 * @return the name of the task that the @p svm solves (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view svm_type_to_task_name(svm_type svm) noexcept;

/**
 * @brief Determine the SVM type used in the model file @p filename.
 * @details The @p filename is assumed to be a valid LIBSVM model file.
 * @param[in] filename the model file name
 * @throws plssvm::invalid_file_format_exception if "svm_type" or "SV" are missing in the LIBSVM model header
 * @return the C-SVM type used to train the model in @p filename (`[[nodiscard]]`)
 */
[[nodiscard]] svm_type svm_type_from_model_file(const std::string &filename);

/**
 * @brief Use the input-stream @p in to initialize the @p svm type.
 * @param[in,out] in input-stream to extract the SVM type from
 * @param[in] svm the SVM type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, svm_type &svm);

}  // namespace plssvm

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::svm_type> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_SVM_TYPES_HPP_
