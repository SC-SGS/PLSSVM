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

#pragma once

#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::split_as
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::invalid_file_format_exception

#include "fmt/compile.h" // FMT_COMPILE
#include "fmt/format.h"  // fmt::format
#include "fmt/os.h"      // fmt::ostream

#include <string_view>  // std::string_view
#include <utility>      // std::pair, std::make_pair
#include <vector>       // std::vector

namespace plssvm::detail::io {

template <typename real_type, typename factors_type>
void read_scaling_factors(file_reader &reader, std::pair<real_type, real_type> &scaling_interval, std::vector<factors_type> &scaling_factors) {
    // at least two lines ("x" + scale interval)
    if (reader.num_lines() < 2) {
        throw invalid_file_format_exception{ fmt::format("At least two lines must be present, but only {} were given!", reader.num_lines()) };
    }

    // discard first line
    // second line contains the scaling range
    const std::vector<real_type> scale_to_interval = detail::split_as<real_type>(reader.line(1));
    if (scale_to_interval.size() != 2) {
        throw invalid_file_format_exception{ fmt::format("The interval to which the data points should be scaled can only contain two values, but {} were given!", scale_to_interval.size()) };
    }
    scaling_interval = std::make_pair(scale_to_interval[0], scale_to_interval[1]);

    // parse scaling factors
    scaling_factors.resize(reader.num_lines() - 2);
    #pragma omp parallel for default(none) shared(scaling_factors, reader)
    for (typename std::vector<factors_type>::size_type i = 0; i < scaling_factors.size(); ++i) {
        const std::string_view line = reader.line(i + 2);
        const std::vector<real_type> values = detail::split_as<real_type>(line);
        if (values.size() != 3) {
            throw invalid_file_format_exception{ fmt::format("Each line must exactly contain three values, but {} were given!", values.size()) };
        }
        // ignore first value
        scaling_factors[i].feature = static_cast<decltype(scaling_factors[i].feature)>(values[0]) - 1;
        scaling_factors[i].lower = values[1];
        scaling_factors[i].upper = values[2];
    }
}

template <typename real_type, typename factors_type>
void write_scaling_factors(fmt::ostream &out, const std::pair<real_type, real_type> &scaling_interval, const std::vector<factors_type> &scaling_factors) {
    out.print("x\n");
    out.print("{} {}\n", scaling_interval.first, scaling_interval.second);
    for (const factors_type &f : scaling_factors) {
        out.print(FMT_COMPILE("{} {} {}\n"), f.index + 1, f.lower, f.upper);
    }
}

}