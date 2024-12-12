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

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

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
