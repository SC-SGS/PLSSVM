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

#pragma once

#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::invalid_file_format_exception

#include "fmt/compile.h" // FMT_COMPILE
#include "fmt/format.h"  // fmt::format, fmt::format_to
#include "fmt/os.h"      // fmt::ostream

#include <algorithm>    // std::max, std::min
#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::exception_ptr, std::current_exception, std::rethrow_exception
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector


namespace plssvm::detail::io {

std::size_t parse_libsvm_num_features(file_reader &reader, const std::size_t num_data_points, const std::size_t start) {
    std::size_t num_features = 0;
    std::exception_ptr parallel_exception;

    #pragma omp parallel default(none) shared(reader, parallel_exception, num_features) firstprivate(num_data_points, start)
    {
        #pragma omp for reduction(max : num_features)
        for (std::size_t i = 0; i < num_data_points; ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = reader.line(i + start);

                // check index of last feature entry
                std::string_view::size_type pos_colon = line.find_last_of(':');
                std::string_view::size_type pos_whitespace = line.find_last_of( ' ', pos_colon);
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

    // no features were parsed -> invalid file
    if (num_features == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file: no data points are given!") };
    }

    return num_features;
}

template <typename real_type, typename label_type>
bool read_libsvm_data(file_reader &reader, const std::size_t start, std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, std::shared_ptr<std::vector<label_type>> &y_ptr) {
    std::exception_ptr parallel_exception;
    bool has_label = true;

    #pragma omp parallel default(none) shared(reader, start, X_ptr, y_ptr, parallel_exception, has_label)
    {
        #pragma omp for reduction(&& : has_label)
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < X_ptr->size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = reader.line(i + start);

                // check if class labels are present (not necessarily the case for test files)
                std::string_view::size_type pos = line.find_first_of(" \n");
                std::string_view::size_type first_colon = line.find_first_of(":\n");
                if (first_colon >= pos) {
                    // get class or alpha
                    (*y_ptr)[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(0, pos));
                } else {
                    has_label = false;
                    pos = 0;
                }

                // get data
                while (true) {
                    std::string_view::size_type next_pos = line.find_first_of(':', pos);
                    // no further data points
                    if (next_pos == std::string_view::npos) {
                        break;
                    }

                    // get index
                    const auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos)) - 1;
                    if (index >= 200) {
                        fmt::print("{} ", index);
                    }
                    if (i >= 500) {
                        fmt::print("{} ", i);
                    }
                    pos = next_pos + 1;

                    // get value
                    next_pos = line.find_first_of(' ', pos);
                    (*X_ptr)[i][index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                    pos = next_pos;
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

    return has_label;
}

template <typename real_type, typename label_type, bool has_label>
void write_libsvm_data(fmt::ostream &out, const std::shared_ptr<std::vector<std::vector<real_type>>> &X_ptr, const std::shared_ptr<std::vector<label_type>> &y_ptr = nullptr) {
    if constexpr (has_label) {
        PLSSVM_ASSERT(y_ptr != nullptr, "has_label is 'true' but no labels were provided!");
    }

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
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:e} "), j + i + 1, d[j + i]);
                }
            }
            output.append(buffer, ptr - buffer);
        }
        output.push_back('\n');
    };

    #pragma omp parallel default(none) shared(out, X_ptr, y_ptr, format_libsvm_line, has_label)
    {
        // all support vectors with class 1
        std::string out_string;
        #pragma omp for schedule(dynamic) nowait
        for (typename std::vector<real_type>::size_type i = 0; i < y_ptr->size(); ++i) {
            if constexpr (has_label) {
                out_string.append(fmt::format(FMT_COMPILE("{} "), (*y_ptr)[i]));
            }
            format_libsvm_line(out_string, (*y_ptr)[i], (*X_ptr)[i]);
        }

        #pragma omp critical
        out.print("{}", out_string);
    }
}

}