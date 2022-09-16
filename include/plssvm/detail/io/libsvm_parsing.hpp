/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the LIBSVM file format.
 */

#ifndef PLSSVM_DETAIL_IO_LIBSVM_PARSING_HPP_
#define PLSSVM_DETAIL_IO_LIBSVM_PARSING_HPP_
#pragma once

#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::invalid_file_format_exception
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format, fmt::format_to
#include "fmt/os.h"       // fmt::ostream

#include <algorithm>    // std::max, std::min
#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::exception_ptr, std::current_exception, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm::detail::io {

[[nodiscard]] inline std::size_t parse_libsvm_num_features(const std::vector<std::string_view> &lines, const std::size_t skipped_lines = 0) {
    std::size_t num_features = 0;
    std::exception_ptr parallel_exception;

#pragma omp parallel default(none) shared(lines, parallel_exception, num_features) firstprivate(skipped_lines)
    {
#pragma omp for reduction(max \
                          : num_features)
        for (std::size_t i = skipped_lines; i < lines.size(); ++i) {
#pragma omp cancellation point for
            try {
                const std::string_view line = lines[i];

                // check index of last feature entry
                const std::string_view::size_type pos_colon = line.find_last_of(':');
                if (pos_colon == std::string_view::npos) {
                    // no features could be found -> can't contribute to the number of feature calculation
                    continue;
                }
                std::string_view::size_type pos_whitespace = line.find_last_of(' ', pos_colon);
                if (pos_whitespace == std::string_view::npos) {
                    // no whitespace BEFORE the last colon could be found
                    // this may only happen if NO labels are given
                    pos_whitespace = 0;
                }
                const auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos_whitespace, pos_colon - pos_whitespace));
                num_features = std::max<std::size_t>(num_features, index);
            } catch (const std::exception &) {
// catch first exception and store it
#pragma omp critical
                {
                    if (!parallel_exception) {
                        parallel_exception = std::current_exception();
                    }
                }
// cancel parallel execution, needs env variable OMP_CANCELLATION=true
#pragma omp cancel for
            }
        }
    }

    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }

    return num_features;
}

template <typename real_type, typename label_type>
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, std::vector<std::vector<real_type>>, std::vector<label_type>> parse_libsvm_data(file_reader &reader, const std::size_t skipped_lines = 0) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");
    // sanity check: can't skip more lines than are present
    PLSSVM_ASSERT(skipped_lines <= reader.num_lines(), "Tried to skipp {} lines, but only {} are present!", skipped_lines, reader.num_lines());

    // parse sizes
    const std::size_t num_data_points = reader.num_lines() - skipped_lines;
    const std::size_t num_features = parse_libsvm_num_features(reader.lines(), skipped_lines);

    // no features were parsed -> invalid file
    if (num_features == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file: no data points are given!") };
    }

    // create vector containing the data and label
    std::vector<std::vector<real_type>> data(num_data_points);
    std::vector<label_type> label(num_data_points);

    std::exception_ptr parallel_exception;
    bool has_label = false;
    bool has_no_label = false;

#pragma omp parallel default(none) shared(reader, skipped_lines, data, label, parallel_exception, has_label, has_no_label) firstprivate(num_features)
    {
#pragma omp for reduction(||                        \
                          : has_label) reduction(|| \
                                                 : has_no_label)
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size(); ++i) {
#pragma omp cancellation point for
            try {
                std::string_view line = reader.line(skipped_lines + i);
                unsigned long last_index = 0;

                // check if class labels are present (not necessarily the case for test files)
                std::string_view::size_type pos = line.find_first_of(" \n");
                std::string_view::size_type first_colon = line.find_first_of(":\n");
                if (first_colon >= pos) {
                    // get class or alpha
                    has_label = true;
                    label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(0, pos));
                } else {
                    has_no_label = true;
                    pos = 0;
                }

                // get data
                std::vector<real_type> vline(num_features);
                while (true) {
                    std::string_view::size_type next_pos = line.find_first_of(':', pos);
                    // no further data points
                    if (next_pos == std::string_view::npos) {
                        break;
                    }

                    // get index
                    auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));

                    // LIBSVM assumes a 1-based indexing -> if the parsed index is 0 this condition is violated
                    if (index == 0) {
                        throw invalid_file_format_exception{ "LIBSVM assumes a 1-based feature indexing scheme, but 0 was given!" };
                    }
                    // the indices must be strictly increasing!
                    if (last_index >= index) {
                        throw invalid_file_format_exception{ fmt::format("The features indices must be strictly increasing, but {} is smaller or equal than {}!", index, last_index) };
                    }
                    last_index = index;

                    // since arrays start at 0, reduce 1 based index by one
                    --index;
                    pos = next_pos + 1;

                    // get value
                    next_pos = line.find_first_of(' ', pos);
                    vline[index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                    pos = next_pos;
                }
                // move filled line to overall matrix
                data[i] = std::move(vline);
            } catch (const std::exception &) {
// catch first exception and store it
#pragma omp critical
                {
                    if (!parallel_exception) {
                        parallel_exception = std::current_exception();
                    }
                }
// cancel parallel execution, needs env variable OMP_CANCELLATION=true
#pragma omp cancel for
            }
        }
    }

    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }
    if (has_label && has_no_label) {
        // some data points where given with labels, BUT some data pints where given without labels
        throw invalid_file_format_exception{ "Inconsistent label specification found (some data points are labeled, others are not)!" };
    }

    return std::make_tuple(num_data_points, num_features, std::move(data), !has_no_label ? std::move(label) : std::vector<label_type>{});
}

template <typename real_type, typename label_type, bool has_label>
inline void write_libsvm_data_impl(const std::string &filename, const std::vector<std::vector<real_type>> &X, const std::vector<label_type> &y) {
    if constexpr (has_label) {
        PLSSVM_ASSERT(X.empty() || !y.empty(), "has_label is 'true' but no labels were provided!");
        PLSSVM_ASSERT(X.size() == y.size(), "Number of data points ({}) and number of labels ({}) mismatch!", X.size(), y.size());
    }

    // create output file
    fmt::ostream out = fmt::output_file(filename);

    // format one output-line
    auto format_libsvm_line = [](std::string &output, const std::vector<real_type> &d) {
        static constexpr std::size_t BLOCK_SIZE = 64;
        static constexpr std::size_t CHARS_PER_BLOCK = 128;
        static constexpr std::size_t BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static char buffer[BUFFER_SIZE];
#pragma omp threadprivate(buffer)

        for (typename std::vector<real_type>::size_type j = 0; j < d.size(); j += BLOCK_SIZE) {
            char *ptr = buffer;
            for (std::size_t i = 0; i < std::min<std::size_t>(BLOCK_SIZE, d.size() - j); ++i) {
                if (d[j + i] != real_type{ 0.0 }) {
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:.10e} "), j + i + 1, d[j + i]);
                }
            }
            output.append(buffer, ptr - buffer);
        }
        output.push_back('\n');
    };

#pragma omp parallel default(none) shared(out, X, y, format_libsvm_line)
    {
        // all support vectors
        std::string out_string;
#pragma omp for schedule(dynamic) nowait
        for (typename std::vector<real_type>::size_type i = 0; i < X.size(); ++i) {
            if constexpr (has_label) {
                out_string.append(fmt::format(FMT_COMPILE("{} "), y[i]));
            }
            format_libsvm_line(out_string, X[i]);
        }

#pragma omp critical
        out.print("{}", out_string);
    }
}

template <typename real_type, typename label_type>
inline void write_libsvm_data(const std::string &filename, const std::vector<std::vector<real_type>> &X, const std::vector<label_type> &y) {
    write_libsvm_data_impl<real_type, label_type, true>(filename, X, y);
}

template <typename real_type>
inline void write_libsvm_data(const std::string &filename, const std::vector<std::vector<real_type>> &X) {
    write_libsvm_data_impl<real_type, real_type, false>(filename, X, {});
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_LIBSVM_PARSING_HPP_