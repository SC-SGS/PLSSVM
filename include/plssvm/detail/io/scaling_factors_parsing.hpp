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

#include "plssvm/constants.hpp"                 // plssvm::real_type
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::split_as
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::trim
#include "plssvm/detail/utility.hpp"            // plssvm:detail::current_date_time
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
 * @brief Read the scaling interval and factors stored using LIBSVM's file format from the file @p filename.
 * @details An example file can look like
 * @code
 * x
 * -1 1
 * 1 -1.117827500607882 1.88494043717792
 * 2 -2.908718888125099 1.0140559662470605
 * 3 -0.13086851759108944 0.6663834427003914
 * 4 0.10805254527169827 1.6464627048813514
 * @endcode
 * Note that the scaling factors are given using an one-based indexing scheme, but are internally stored using zero-based indexing.
 * @tparam factors_type plssvm::data_set<real_type>::scaling::factors (cannot be forward declared or included)
 * @param[in] reader the file_reader used to read the scaling factors
 * @throws plssvm::invalid_file_format_exception if the header is omitted ('x' and the scaling interval)
 * @throws plssvm::invalid_file_format_exception if the first line doesn't only contain `x`
 * @throws plssvm::invalid_file_format_exception if the scaling interval is provided with more or less than two values
 * @throws plssvm::invalid_file_format_exception if the scaling factors are provided with more or less than three values
 * @throws plssvm::invalid_file_format_exception if the scaling factors feature index is zero-based instead of one-based
 * @return the read scaling interval and factors (`[[nodiscard]]`)
 */
template <typename factors_type>
[[nodiscard]] inline std::tuple<std::pair<real_type, real_type>, std::vector<factors_type>> parse_scaling_factors(const file_reader &reader) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");

    // at least two lines ("x" + scale interval)
    if (reader.num_lines() < 2) {
        throw invalid_file_format_exception{ fmt::format("At least two lines must be present, but only {} were given!", reader.num_lines()) };
    }

    // first line must always contain the single character 'x'
    if (detail::trim(reader.line(0)) != "x") {
        throw invalid_file_format_exception{ fmt::format("The first line must only contain an 'x', but is \"{}\"!", reader.line(0)) };
    }
    // second line contains the scaling range
    std::vector<real_type> scale_to_interval = detail::split_as<real_type>(reader.line(1));
    if (scale_to_interval.size() != 2) {
        throw invalid_file_format_exception{ fmt::format("The interval to which the data points should be scaled must exactly have two values, but {} were given!", scale_to_interval.size()) };
    }
    if (scale_to_interval[0] >= scale_to_interval[1]) {
        throw invalid_file_format_exception{ fmt::format("Inconsistent scaling interval specification: lower ({}) must be less than upper ({})!", scale_to_interval[0], scale_to_interval[1]) };
    }

    // parse scaling factors
    std::exception_ptr parallel_exception;
    std::vector<factors_type> scaling_factors(reader.num_lines() - 2);
    #pragma omp parallel default(none) shared(parallel_exception, scaling_factors, reader)
    {
        #pragma omp for
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
                const auto feature = static_cast<decltype(scaling_factors[i].feature)>(values[0]);
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
    }
    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }

    return std::make_tuple(std::make_pair(scale_to_interval[0], scale_to_interval[1]), std::move(scaling_factors));
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
 * @tparam factors_type plssvm::data_set<real_type>::scaling::factors (cannot be forward declared or included)
 * @param[in] filename the filename to write the data to
 * @param[in] scaling_interval the valid scaling interval, i.e., [first, second]
 * @param[in] scaling_factors the scaling factor for each feature; given **zero** based, but written to file **one** based!
 */
template <typename factors_type>
inline void write_scaling_factors(const std::string &filename, const std::pair<real_type, real_type> &scaling_interval, const std::vector<factors_type> &scaling_factors) {
    PLSSVM_ASSERT(scaling_interval.first < scaling_interval.second, "Illegal interval specification: lower ({}) < upper ({}).", scaling_interval.first, scaling_interval.second);

    // create output file
    fmt::ostream out = fmt::output_file(filename);
    // write timestamp as current date time
    out.print("# These scaling factors have been created at {}\n", detail::current_date_time());

    // x must always be outputted
    out.print("x\n");
    // write the requested scaling interval
    out.print("{} {}\n", scaling_interval.first, scaling_interval.second);
    // write the scaling factors for each feature, note the one based indexing scheme!
    for (const factors_type &f : scaling_factors) {
        out.print("{} {:.10e} {:.10e}\n", f.feature + 1, f.lower, f.upper);
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_SCALING_FACTORS_PARSING_HPP_