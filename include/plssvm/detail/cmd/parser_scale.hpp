/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a class encapsulating all necessary parameters for scaling a data set possibly provided through command line arguments.
 */

#ifndef PLSSVM_DETAIL_CMD_PARSER_SCALE_HPP_
#define PLSSVM_DETAIL_CMD_PARSER_SCALE_HPP_
#pragma once

#include "plssvm/constants.hpp"          // plssvm::real_type
#include "plssvm/file_format_types.hpp"  // plssvm::file_format_type

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream
#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
 * @brief Struct for encapsulating all necessary parameters for scaling a data set; normally provided through command line arguments.
 */
struct parser_scale {
    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the scale parameters accordingly.
     * @details If no scaled filename is given, the scaled data is directly output to the terminal (the default behavior of LIBSVM).
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parser_scale(int argc, char **argv);

    /// The lower bound of the scaled data values.
    real_type lower{ -1.0 };
    /// The upper bound of the scaled data values.
    real_type upper{ +1.0 };
    /// The file type (currently either LIBSVM or ARFF) to which the scaled data should be written to.
    file_format_type format{ file_format_type::libsvm };

    /// `true` if `std::string` should be used as label type instead of the default type `ìnt`.
    bool strings_as_labels{ false };

    /// The name of the data file to scale.
    std::string input_filename{};
    /// The name of the scaled data file to save.
    std::string scaled_filename{};
    /// The name of the file where the scaling factors are saved.
    std::string save_filename{};
    /// The name of the file from which the scaling factors should be restored.
    std::string restore_filename{};

    /// If performance tracking has been enabled, provides the name of the file where the performance tracking results are saved to. If the filename is empty, the results are dumped to stdout instead.
    std::string performance_tracking_filename{};
};

/**
 * @brief Output all scale parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the scale parameters to
 * @param[in] params the scale parameters
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parser_scale &params);

}  // namespace plssvm::detail::cmd

template <>
struct fmt::formatter<plssvm::detail::cmd::parser_scale> : fmt::ostream_formatter {};

#endif  // PLSSVM_DETAIL_CMD_PARSER_SCALE_HPP_