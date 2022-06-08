/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the ARFF file format.
 */

#pragma once

#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/operators.hpp"          // plssvm::operator::sign
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::to_upper_case, plssvm::detail::starts_with
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception::invalid_file_format_exception

#include "fmt/format.h"   // fmt::format, fmt::join
#include "fmt/os.h"       // fmt::ostream

#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::exception_ptr, std::current_exception, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <vector>       // std::vector


namespace plssvm::detail::io {


inline std::tuple<std::size_t, std::size_t, bool> read_arff_header(file_reader &reader) {
    std::size_t num_features = 0;
    bool has_label = false;

    // parse arff header
    std::size_t header_line = 0;
    for (; header_line < reader.num_lines(); ++header_line) {
        std::string line{ reader.line(header_line) };
        detail::to_upper_case(line);
        if (detail::starts_with(line, "@RELATION")) {
            // ignore relation
            continue;
        } else if (detail::starts_with(line, "@ATTRIBUTE")) {
            if (line.find("CLASS") != std::string::npos) {
                if (has_label) {
                    // only one class attribute is allowed
                    throw invalid_file_format_exception{ "Only the last ATTRIBUTE may be CLASS!" };
                }
                // found a class
                has_label = true;
                continue; // don't increment num_features
            } else if (line.find("NUMERIC") == std::string::npos) {
                throw invalid_file_format_exception{ fmt::format("Can only use NUMERIC features, but '{}' was given!", reader.line(header_line)) };
            }
            // add a feature
            ++num_features;
        } else if (detail::starts_with(line, "@DATA")) {
            // finished reading header -> start parsing data
            break;
        }
    }

    // perform additional checks
    if (num_features == 0) {
        // no @ATTRIBUTE fields
        throw invalid_file_format_exception{ "Can't parse file: no ATTRIBUTES are defined!" };
    } else if (header_line + 1 >= reader.num_lines()) {
        // no data points provided
        throw invalid_file_format_exception{ "Can't parse file: no data points are given or @DATA is missing!" };
    }

    return std::make_tuple(header_line, num_features, has_label);
}

template <typename real_type, typename label_type>
inline void read_arff_data(file_reader &reader, std::size_t header, std::size_t num_features, std::size_t max_size, bool has_label, std::vector<std::vector<real_type>> &X, std::vector<label_type> &y) {
    std::exception_ptr parallel_exception;

    #pragma omp parallel default(none) shared(reader, X, y, parallel_exception) firstprivate(header, num_features, max_size, has_label)
    {
        #pragma omp for
        for (std::size_t i = 0; i < X.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = reader.line(i + header + 1);
                //
                if (detail::starts_with(line, '@')) {
                    // read @ inside data section
                    throw invalid_file_format_exception{ fmt::format("Read @ inside data section!: '{}'", line) };
                }

                // parse sparse or dense data point definition
                if (detail::starts_with(line, '{')) {
                    // missing closing }
                    if (!detail::ends_with(line, '}')) {
                        throw invalid_file_format_exception{ fmt::format("Missing closing '}}' for sparse data point {} description!", i) };
                    }
                    // sparse line
                    bool is_class_set = false;
                    std::string_view::size_type pos = 1;
                    while (true) {
                        std::string_view::size_type next_pos = line.find_first_of(' ', pos);
                        // no further data points
                        if (next_pos == std::string_view::npos) {
                            break;
                        }

                        // get index
                        const auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        if (index >= max_size) {
                            // index too big for specified number of features
                            throw invalid_file_format_exception{ fmt::format("Too many features given! Trying to add feature at position {} but max position is {}!", index, num_features - 1) };
                        }
                        pos = next_pos + 1;

                        // get value
                        next_pos = line.find_first_of(",}", pos);

                        // write parsed value depending on the index
                        if (index == max_size - 1 && has_label) {
                            is_class_set = true;
                            y[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos));
                        } else {
                            X[i][index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        }

                        // remove already processes part of the line
                        line.remove_prefix(next_pos + 1);
                        line = detail::trim_left(line);
                        pos = 0;
                    }
                    // no class label found
                    if (!is_class_set && has_label) {
                        throw invalid_file_format_exception{ fmt::format("Missing label for data point {}!", i) };
                    }
                } else {
                    // dense line
                    std::string_view::size_type pos = 0;
                    std::string_view::size_type next_pos = 0;
                    for (std::size_t j = 0; j < max_size - 1; ++j) {
                        next_pos = line.find_first_of(',', pos);
                        if (next_pos == std::string_view::npos) {
                            throw invalid_file_format_exception{ fmt::format("Invalid number of features/labels! Found {} but should be {}!", j, max_size - 1) };
                        }
                        X[i][j] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;
                    }
                    // write last number to the correct vector (based on the fact whether labels are present or not)
                    if (has_label) {
                        y[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos));
                    } else {
                        X[i][num_features - 1] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos));
                    }
                    // check whether superfluous data points are left
                    next_pos = line.find_first_of(',', pos);
                    if (next_pos != std::string_view::npos) {
                        throw invalid_file_format_exception{ fmt::format("Too many features! Superfluous '{}' for data point {}!", line.substr(next_pos), i) };
                    }
                }
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
}

template <typename label_type>
inline void write_arff_header(fmt::ostream &out, const std::size_t num_features, const bool has_labels) {
    for (std::size_t i = 0; i < num_features; ++i) {
        out.print("@ATTRIBUTE feature{} NUMERIC\n", i);
    }
    if (has_labels) {
        if constexpr (std::is_same_v<detail::remove_cvref_t<label_type>, std::string>) {
            out.print("@ATTRIBUTE class STRING\n\n");
        } else {
            out.print("@ATTRIBUTE class NUMERIC\n\n");
        }
    }
    out.print("@DATA\n");
}

template <typename real_type, typename label_type>
inline void write_arff_data(fmt::ostream &out, const std::vector<std::vector<real_type>> &X, const std::vector<label_type> &y) {
    #pragma omp parallel default(none) shared(out, X, y)
    {
        // all support vectors
        std::string out_string;
        #pragma omp for schedule(dynamic) nowait
        for (std::size_t i = 0; i < X.size(); ++i) {
            out_string.append(fmt::format("{},{}\n", fmt::join(X[i], ","), y[i]));
        }

        #pragma omp critical
        out.print("{}", out_string);
    }
}

template <typename real_type>
inline void write_arff_data(fmt::ostream &out, const std::vector<std::vector<real_type>> &X) {
    #pragma omp parallel default(none) shared(out, X)
    {
        // all support vectors
        std::string out_string;
        #pragma omp for schedule(dynamic) nowait
        for (std::size_t i = 0; i < X.size(); ++i) {
            out.print("{}\n", fmt::join(X[i], ","));
        }

        #pragma omp critical
        out.print("{}", out_string);
    }
}

}