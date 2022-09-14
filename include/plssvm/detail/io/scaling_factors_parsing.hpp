/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the scaling factor file parsing.
 */

#ifndef PLSSVM_DETAIL_IO_SCALING_FACTORS_PARSING_HPP_
#define PLSSVM_DETAIL_IO_SCALING_FACTORS_PARSING_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::split_as
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::trim
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::invalid_file_format_exception

#include "fmt/core.h"  // fmt::format
#include "fmt/os.h"    // fmt::ostream, fmt::output_file

#include <exception>    // std::exception_ptr, std::exception, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Read the @p scaling_interval and @p scaling_factors stored using LIBSVM's file format from the file @p filename.
 * @details An example file can look like
 * @code
 * x
 * -1 1
 * 1 -1.117827500607882 1.88494043717792
 * 2 -2.908718888125099 1.0140559662470605
 * 3 -0.13086851759108944 0.6663834427003914
 * 4 0.10805254527169827 1.6464627048813514
 * @endcode
 * @tparam real_type the used floating point type
 * @tparam factors_type plssvm::data_set<real_type>::scaling::factors (cannot be forward declared or included)
 * @param[in] filename the filename to read the data from
 * @param[out] scaling_interval the output parameter to which the scaling interval will be written
 * @param[out] scaling_factors the output parameter to which the scaling factors will be written; given **zero** based, but returned **one** based!
 */
template <typename real_type, typename factors_type>
void read_scaling_factors(const std::string &filename, std::pair<real_type, real_type> &scaling_interval, std::vector<factors_type> &scaling_factors) {
    PLSSVM_ASSERT(scaling_factors.empty(), "{} scaling factors are already provided!", scaling_factors.size());

    // open file and parse content
    plssvm::detail::io::file_reader reader{ filename };
    reader.read_lines('#');

    // at least two lines ("x" + scale interval)
    if (reader.num_lines() < 2) {
        throw invalid_file_format_exception{ fmt::format("At least two lines must be present, but only {} were given!", reader.num_lines()) };
    }

    // first line must always contain the single character 'x'
    if (detail::trim(reader.line(0)) != "x") {
        throw invalid_file_format_exception{ fmt::format("The first line must only contain an 'x', but is \"{}\"!", reader.line(0)) };
    }
    // second line contains the scaling range
    const std::vector<real_type> scale_to_interval = detail::split_as<real_type>(reader.line(1));
    if (scale_to_interval.size() != 2) {
        throw invalid_file_format_exception{ fmt::format("The interval to which the data points should be scaled must exactly have two values, but {} were given!", scale_to_interval.size()) };
    }
    scaling_interval = std::make_pair(scale_to_interval[0], scale_to_interval[1]);

    // parse scaling factors
    std::exception_ptr parallel_exception;
    scaling_factors.resize(reader.num_lines() - 2);
    #pragma omp parallel for default(none) shared(parallel_exception, scaling_factors, reader)
    for (typename std::vector<factors_type>::size_type i = 0; i < scaling_factors.size(); ++i) {
        try {
            // parse the current line
            const std::string_view line = reader.line(i + 2);
            const std::vector<real_type> values = detail::split_as<real_type>(line);
            // check if the line contains the correct number of values
            if (values.size() != 3) {
                throw invalid_file_format_exception{ fmt::format("Each line must contain exactly three values, but {} were given!", values.size()) };
            }
            // set the scaling factor based on the parsed values
            auto feature = static_cast<decltype(scaling_factors[i].feature)>(values[0]);
            // check if we are one-based, i.e., no 0 must be read as feature value
            if (feature == 0) {
                throw invalid_file_format_exception{ "The scaling factors must be provided one-based, but are zero-based!" };
            }
            scaling_factors[i].feature = feature - 1;
            scaling_factors[i].lower = values[1];
            scaling_factors[i].upper = values[2];
        } catch (const std::exception &) {
            // catch first exception and store it
            #pragma omp critical
            {
                if (!parallel_exception) {
                    parallel_exception = std::current_exception();
                }
            }
        }
    }
    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }
}

/**
 * @brief Write the @p scaling_interval and @p scaling_factors to a file for later usage in scaling another data set using LIBSVM's file format.
 * @details An example file can look like
 * @code
 * x
 * -1 1
 * 1 -1.117827500607882 1.88494043717792
 * 2 -2.908718888125099 1.0140559662470605
 * 3 -0.13086851759108944 0.6663834427003914
 * 4 0.10805254527169827 1.6464627048813514
 * @endcode
 * @tparam real_type the used floating point type
 * @tparam factors_type plssvm::data_set<real_type>::scaling::factors (cannot be forward declared or included)
 * @param[in] filename the filename to write the data to
 * @param[in] scaling_interval the valid scaling interval, i.e., [first, second]
 * @param[in] scaling_factors the scaling factor for each feature; given **zero** based, but written to file **one** based!
 */
template <typename real_type, typename factors_type>
void write_scaling_factors(const std::string &filename, const std::pair<real_type, real_type> &scaling_interval, const std::vector<factors_type> &scaling_factors) {
    PLSSVM_ASSERT(scaling_interval.first < scaling_interval.second, "Illegal interval specification: lower ({}) < upper ({}).", scaling_interval.first, scaling_interval.second);
    PLSSVM_ASSERT(!scaling_factors.empty(), "No scaling factors provided!");

    // create output file
    fmt::ostream out = fmt::output_file(filename);

    // x must always be outputted
    out.print("x\n");
    // write the requested scaling interval
    out.print("{} {}\n", scaling_interval.first, scaling_interval.second);
    // write the scaling factors for each feature, note the one based indexing scheme!
    for (const factors_type &f : scaling_factors) {
        out.print("{} {} {}\n", f.feature + 1, f.lower, f.upper);
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_SCALING_FACTORS_PARSING_HPP_