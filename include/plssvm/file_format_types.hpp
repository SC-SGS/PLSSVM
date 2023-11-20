/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all supported file formats.
 */

#ifndef PLSSVM_FILE_FORMAT_TYPES_HPP_
#define PLSSVM_FILE_FORMAT_TYPES_HPP_
#pragma once

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm {

/**
 * @brief Enum class for all supported file types.
 */
enum class file_format_type {
    /** The LIBSVM file format. Used as default. For the file format specification see: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html */
    libsvm,
    /** The ARFF file format. For the file format specification see: https://www.cs.waikato.ac.nz/~ml/weka/arff.html */
    arff
};

/**
 * @brief Output the @p format to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the file format type to
 * @param[in] format the file format type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, file_format_type format);

/**
 * @brief Use the input-stream @p in to initialize the @p format type.
 * @param[in,out] in input-stream to extract the file format type from
 * @param[in] format the file format type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, file_format_type &format);

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::file_format_type> : fmt::ostream_formatter {};

#endif  // PLSSVM_FILE_FORMAT_TYPES_HPP_