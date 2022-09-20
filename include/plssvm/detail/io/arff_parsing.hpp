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

#include "fmt/format.h"  // fmt::format, fmt::join
#include "fmt/os.h"      // fmt::ostream

#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::exception_ptr, std::current_exception, std::rethrow_exception
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <vector>       // std::vector

namespace plssvm::detail::io {

[[nodiscard]] inline std::tuple<std::size_t, std::size_t, bool> parse_arff_header(file_reader &reader) {
    std::size_t num_features = 0;
    bool has_label = false;

    const auto check_for_name = [](std::string_view line, const std::size_t prefix, const std::size_t suffix) {
        std::string_view sv{ line };
        sv.remove_prefix(prefix);
        sv.remove_suffix(suffix);
        sv = detail::trim(sv);

        // remaining string may not be empty
        if (sv.empty()) {
            throw invalid_file_format_exception{ fmt::format("The \"{}\" field must contain a name!", line) };
        }
        // check if string contains whitespaces -> must be quoted
        if (detail::contains(sv, ' ') && !detail::starts_with(sv, '"') && !detail::ends_with(sv, '"')) {
            throw invalid_file_format_exception{ fmt::format("A \"{}\" name that contains a whitespace must be quoted!", line) };
        }

        // return name part of line
        return sv;
    };

    // parse arff header
    std::size_t header_line = 0;
    for (; header_line < reader.num_lines(); ++header_line) {
        // get next line and convert content to all upper case
        const std::string_view line{ reader.line(header_line) };
        const std::string upper_case_line = detail::as_upper_case(line);
        // relation fields are ignored
        if (detail::starts_with(upper_case_line, "@RELATION")) {
            // if a relation is given, it must be given in the first line
            if (header_line != 0) {
                throw invalid_file_format_exception{ "The @RELATION attribute must be set before any other @ATTRIBUTE!" };
            }
            // the relation field must contain a name
            check_for_name(line, 9, 0);  // @RELATION is 9 chars long
            // parse next line
            continue;
        }
        // check for attribute fields
        if (detail::starts_with(upper_case_line, "@ATTRIBUTE")) {
            // check if a "normal" numeric feature has been found
            if (upper_case_line.find("NUMERIC") != std::string::npos) {
                // a numeric field must also contain a name
                const std::string_view name = check_for_name(line, 10, 7);  // @ATTRIBUTE is 10 chars long, NUMERIC 7 chars
                // the attribute name "CLASS" is reserved!
                if (detail::as_upper_case(name) == "CLASS") {
                    throw invalid_file_format_exception{ "May not use the combination of the reserved name \"class\" and attribute type NUMERIC!" };
                }
                // add a feature to the running count
                ++num_features;
                continue;
            }

            // only other valid line may be (nominal attribute with the name CLASS)
            // @ATTRIBUTE CLASS {cat,dog}

            // remove attribute from string
            std::string_view sv{ line };
            sv.remove_prefix(10);  // @ATTRIBUTE is 10 chars long
            sv = trim_left(sv);

            // if the line is valid, it must now start with CLASS
            if (detail::starts_with(detail::as_upper_case(sv), "CLASS")) {
                // only one class attribute is allowed
                if (has_label) {
                    throw invalid_file_format_exception{ "A nominal attribute with the name CLASS may only be provided once!" };
                }
                // check if the nominal attribute ist enclosed in curly braces
                sv.remove_prefix(5);  // CLASS is 5 chars long
                sv = detail::trim(sv);
                // the class labels must be given
                if (sv.empty()) {
                    throw invalid_file_format_exception{ fmt::format("The \"{}\" field must contain class labels!", line) };
                }
                // check if string contains whitespaces -> must be quoted
                if (!detail::starts_with(sv, '{') && !detail::ends_with(sv, '}')) {
                    throw invalid_file_format_exception{ fmt::format("The \"{}\" nominal attribute must be enclosed with {{}}!", line) };
                }
                // found a class
                has_label = true;
                continue;  // don't increment num_features
            }
        }
        // check for the data field
        if (detail::starts_with(upper_case_line, "@DATA")) {
            // finished reading header -> start parsing data
            break;
        }
        // check if the line starts with an @ but is not a valid attribute
        if (detail::starts_with(upper_case_line, "@")) {
            // an invalid or unsupported header entry has been read!
            throw invalid_file_format_exception{ fmt::format("Read an invalid header entry: \"{}\"!", line) };
        }
    }

    // perform some additional checks
    if (num_features == 0) {
        // no @ATTRIBUTE fields
        throw invalid_file_format_exception{ "Can't parse file: no feature ATTRIBUTES are defined!" };
    }
    if (header_line + 1 >= reader.num_lines()) {
        // no data points provided
        throw invalid_file_format_exception{ "Can't parse file: @DATA is missing!" };
    }

    return std::make_tuple(num_features, header_line + 1, has_label);
}

template <typename real_type, typename label_type>
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, std::vector<std::vector<real_type>>, std::vector<label_type>> parse_arff_data(file_reader &reader) {
    std::exception_ptr parallel_exception;

    // parse arff header, structured bindings can't be used because of the OpenMP parallel section
    std::size_t header = 0;
    std::size_t max_size = 0;
    bool has_label = false;
    std::tie(max_size, header, has_label) = detail::io::parse_arff_header(reader);

    // calculate data set sizes
    const std::size_t num_data_points = reader.num_lines() - (header);
    const std::size_t num_features = has_label ? max_size - 1 : max_size;

    // create data and label vectors
    std::vector<std::vector<real_type>> data(num_data_points, std::vector<real_type>(num_features));
    std::vector<label_type> label(num_data_points);

#pragma omp parallel default(none) shared(reader, data, label, parallel_exception) firstprivate(header, num_features, max_size, has_label)
    {
#pragma omp for
        for (std::size_t i = 0; i < data.size(); ++i) {
#pragma omp cancellation point for
            try {
                std::string_view line = reader.line(i + header);
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
                            label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos));
                        } else {
                            data[i][index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
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
                        data[i][j] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;
                    }
                    // write last number to the correct vector (based on the fact whether labels are present or not)
                    if (has_label) {
                        label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos));
                    } else {
                        data[i][num_features - 1] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos));
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

    return std::make_tuple(num_data_points, num_features, std::move(data), has_label ? std::move(label) : std::vector<label_type>{});
}

template <typename real_type, typename label_type, bool has_label>
inline void write_arff_data_impl(const std::string &filename, const std::vector<std::vector<real_type>> &data, const std::vector<label_type> &label) {
    // create file
    fmt::ostream out = fmt::output_file(filename);

    const std::size_t num_data_points = data.size();
    if (num_data_points == 0) {
        // nothing to output
        return;
    }
    const std::size_t num_features = data.front().size();

    // write arff header for features
    for (std::size_t i = 0; i < num_features; ++i) {
        out.print("@ATTRIBUTE feature{} NUMERIC\n", i);
    }
    // write arff header for the label if existing
    if constexpr (has_label) {
        if constexpr (std::is_same_v<detail::remove_cvref_t<label_type>, std::string>) {
            out.print("@ATTRIBUTE class STRING\n\n");
        } else {
            out.print("@ATTRIBUTE class NUMERIC\n\n");
        }
    }
    out.print("@DATA\n");

    // write arff data
#pragma omp parallel default(none) shared(out, data, label) firstprivate(num_data_points)
    {
        // all support vectors
        std::string out_string;
#pragma omp for schedule(dynamic) nowait
        for (std::size_t i = 0; i < num_data_points; ++i) {
            if constexpr (has_label) {
                out_string.append(fmt::format("{},{}\n", fmt::join(data[i], ","), label[i]));
            } else {
                out.print("{}\n", fmt::join(data[i], ","));
            }
        }

#pragma omp critical
        out.print("{}", out_string);
    }
}

template <typename real_type, typename label_type>
inline void write_arff_data(const std::string &filename, const std::vector<std::vector<real_type>> &data, const std::vector<label_type> &label) {
    write_arff_data_impl<real_type, label_type, true>(filename, data, label);
}

template <typename real_type>
inline void write_arff_data(const std::string &filename, const std::vector<std::vector<real_type>> &data) {
    write_arff_data_impl<real_type, real_type, false>(filename, data, {});
}

}  // namespace plssvm::detail::io