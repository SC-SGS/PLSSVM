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

#ifndef PLSSVM_DETAIL_IO_ARFF_PARSING_HPP_
#define PLSSVM_DETAIL_IO_ARFF_PARSING_HPP_
#pragma once

#include "plssvm/constants.hpp"                 // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/operators.hpp"          // plssvm::operator::sign
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/detail/string_utility.hpp"     // plssvm::detail::{to_upper_case, as_upper_case, starts_with, ends_with}
#include "plssvm/detail/utility.hpp"            // plssvm::detail::current_date_time
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception::invalid_file_format_exception
#include "plssvm/matrix.hpp"                    // plssvm::soa_matrix

#include "fmt/format.h"  // fmt::format, fmt::join
#include "fmt/os.h"      // fmt::ostream, fmt::output_file

#include <cstddef>      // std::size_t
#include <exception>    // std::exception, std::exception_ptr, std::current_exception, std::rethrow_exception
#include <set>          // std::set
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Parse the ARFF file header, i.e., determine the number of features, the length of the ARFF header, whether the data set is annotated with labels
 *        and at which position the label is written in the data set.
 * @tparam label_type the type of the labels (any arithmetic type or std::string)
 * @param[in] lines the ARFF header to parse
 * @throws plssvm::invalid_file_format_exception if the \@RELATION field does not come before any other \@ATTRIBUTE
 * @throws plssvm::invalid_file_format_exception if the \@RELATION field does not have a name
 * @throws plssvm::invalid_file_format_exception if the \@RELATION field does have a name with whitespaces but is not quoted
 * @throws plssvm::invalid_file_format_exception if an \@ATTRIBUTE field has the type NUMERIC **and** the name CLASS
 * @throws plssvm::invalid_file_format_exception if an \@ATTRIBUTE field does not have a name
 * @throws plssvm::invalid_file_format_exception if an \@ATTRIBUTE field does have a name with whitespaces but is not quoted
 * @throws plssvm::invalid_file_format_exception if multiple \@ATTRIBUTES with the name CLASS are provided
 * @throws plssvm::invalid_file_format_exception if the class field does not provide any labels
 * @throws plssvm::invalid_file_format_exception if the class field provides labels that are no enclosed in {} (ARFF nominal attributes)
 * @throws plssvm::invalid_file_format_exception if only a single label has been provided
 * @throws plssvm::invalid_file_format_exception if a label has been provided multiple times
 * @throws plssvm::invalid_file_format_exception if a string label contains a whitespace
 * @throws plssvm::invalid_file_format_exception if a header entry starts with an @ but is none of \@RELATION, \@ATTRIBUTE, or \@DATA
 * @throws plssvm::invalid_file_format_exception if no feature attributes are provided
 * @throws plssvm::invalid_file_format_exception if the \@DATA attribute is missing
 * @return the necessary header information: [num_features, num_header_lines, unique_labels, label_idx] (`[[nodiscard]]`)
 */
template <typename label_type>
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, std::set<label_type>, std::size_t> parse_arff_header(const std::vector<std::string_view> &lines) {
    std::size_t num_features = 0;
    std::size_t label_idx = 0;
    bool has_label = false;
    std::set<label_type> labels{};

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
    for (; header_line < lines.size(); ++header_line) {
        // get next line and convert content to all upper case
        const std::string_view line = lines[header_line];
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
                // increment class index as long as no class labels have been read
                if (!has_label) {
                    ++label_idx;
                }
                continue;
            }

            // only other valid line may be (nominal attribute with the name CLASS)
            // @ATTRIBUTE CLASS {cat,dog}

            // remove attribute from string
            std::string_view sv{ line };
            sv.remove_prefix(std::string_view{ "@ATTRIBUTE" }.size());
            sv = trim_left(sv);

            // if the line is valid, it must now start with CLASS
            if (detail::starts_with(detail::as_upper_case(sv), "CLASS")) {
                // only one class attribute is allowed
                if (has_label) {
                    throw invalid_file_format_exception{ "A nominal attribute with the name CLASS may only be provided once!" };
                }
                // check if the nominal attribute ist enclosed in curly braces
                sv.remove_prefix(std::string_view{ "CLASS" }.size());
                sv = detail::trim(sv);
                // the class labels must be given
                if (sv.empty()) {
                    throw invalid_file_format_exception{ fmt::format("The \"{}\" field must contain class labels!", line) };
                }
                // check if string contains whitespaces -> must be quoted
                if (!detail::starts_with(sv, '{') && !detail::ends_with(sv, '}')) {
                    throw invalid_file_format_exception{ fmt::format("The \"{}\" nominal attribute must be enclosed with {{}}!", line) };
                }
                // remove curly braces
                sv = sv.substr(1, sv.size() - 2);
                // split string with delimiter ',' to check the number of provided classes
                const std::vector<std::string_view> labels_split = detail::split(sv, ',');
                if (labels_split.size() == 1) {
                    throw invalid_file_format_exception{ "Only a single label has been provided!" };
                }
                // check whether only unique labels have been provided
                for (const std::string_view label : labels_split) {
                    labels.insert(detail::convert_to<label_type, invalid_file_format_exception>(detail::trim(label)));
                }
                if (labels_split.size() != labels.size()) {
                    throw invalid_file_format_exception{ fmt::format("Provided {} labels but only {} of them was/where unique!", labels_split.size(), labels.size()) };
                }
                // check whether a string label contains a whitespace
                if constexpr (std::is_same_v<label_type, std::string>) {
                    for (const std::string_view label : labels_split) {
                        if (detail::contains(detail::trim(label), ' ')) {
                            throw invalid_file_format_exception{ fmt::format("String labels may not contain whitespaces, but \"{}\" has at least one!", detail::trim(label)) };
                        }
                    }
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
    if (header_line + 1 >= lines.size()) {
        // no data points provided
        throw invalid_file_format_exception{ "Can't parse file: @DATA is missing!" };
    }

    return std::make_tuple(num_features, header_line + 1, labels, has_label ? label_idx : 0);
}

/**
 * @brief Parse all data points and potential label using the file @p reader, ignoring all empty lines and lines starting with an `%`.
 *        If no labels are found, returns an empty vector.
 * @details An example file can look like
 * @code
 * @RELATION name
 *
 * @ATTRIBUTE feature_0   numeric
 * @ATTRIBUTE feature_1   numeric
 * @ATTRIBUTE feature_2   numeric
 * @ATTRIBUTE feature_3   numeric
 * @ATTRIBUTE class       {-1,1}
 *
 * @DATA
 * -1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1
 * -0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026,1
 * 0.0,0.60276937379453293,-0.13086851759108944,0.0,-1
 * 0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387,-1
 * 1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514,-1
 * @endcode
 * @tparam label_type the type of the labels (any arithmetic type or std::string)
 * @param[in] reader the file_reader used to read the ARFF data
 * @note The features must be provided with zero-based indices!
 * @throws plssvm::invalid_file_format_exception if no features could be found (may indicate an empty file)
 * @throws plssvm::invalid_file_format_exception if a label couldn't be converted to the provided @p label_type
 * @throws plssvm::invalid_file_format_exception if a feature index couldn't be converted to `unsigned long`
 * @throws plssvm::invalid_file_format_exception if a feature value couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if an '@' is read inside the \@DATA section
 * @throws plssvm::invalid_file_format_exception if a closing curly brace '}' is missing in the sparse data point description
 * @throws plssvm::invalid_file_format_exception if an closing curly brace '{' is missing in the sparse data point description
 * @throws plssvm::invalid_file_format_exception if a index is out-of-bounce with respect to the provided ARFF header information
 * @throws plssvm::invalid_file_format_exception if the ARFF header specifies labels but any data point misses a label
 * @throws plssvm::invalid_file_format_exception if the number of found features and labels mismatches the numbers provided in the ARFF header
 * @throws plssvm::invalid_file_format_exception if a label in the data section has been found, that did not appear in the header
 * @return a std::tuple containing: [the number of data points, the number of features per data point, the data points, the labels (optional)] (`[[nodiscard]]`)
 */
template <typename label_type>
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, soa_matrix<real_type>, std::vector<label_type>> parse_arff_data(const file_reader &reader) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");

    // parse arff header, structured bindings can't be used because of the OpenMP parallel section
    std::size_t num_header_lines = 0;
    std::size_t num_features = 0;
    std::set<label_type> unique_label{};
    std::size_t label_idx = 0;
    std::tie(num_features, num_header_lines, unique_label, label_idx) = detail::io::parse_arff_header<label_type>(reader.lines());
    const bool has_label = !unique_label.empty();

    // calculate data set sizes
    const std::size_t num_data_points = reader.num_lines() - num_header_lines;
    const std::size_t num_attributes = num_features + static_cast<std::size_t>(has_label);

    // create data and label vectors
    soa_matrix<real_type> data{ num_data_points, num_features, PADDING_SIZE, PADDING_SIZE };
    std::vector<label_type> label(num_data_points);

    std::exception_ptr parallel_exception;

    #pragma omp parallel default(none) shared(reader, data, label, unique_label, parallel_exception) firstprivate(num_header_lines, num_data_points, num_attributes, has_label, label_idx)
    {
        #pragma omp for
        for (std::size_t i = 0; i < num_data_points; ++i) {
            try {
                std::string_view line = reader.line(i + num_header_lines);
                // there must not be any @ inside the data section
                if (detail::starts_with(line, '@')) {
                    throw invalid_file_format_exception{ fmt::format("Read @ inside data section!: \"{}\"!", line) };
                }

                // parse sparse or dense data point definition
                // a sparse data point must start with a opening curly brace
                if (detail::starts_with(line, '{')) {
                    // -> sparse data point given, but the closing brace is missing
                    if (!detail::ends_with(line, '}')) {
                        throw invalid_file_format_exception{ fmt::format("Missing closing '}}' for sparse data point \"{}\" description!", line) };
                    }
                    // parse the sparse line
                    bool is_class_set = false;
                    std::string_view::size_type pos = 1;
                    while (true) {
                        std::string_view::size_type next_pos = line.find_first_of(' ', pos);
                        // no further data points
                        if (next_pos == std::string_view::npos) {
                            break;
                        }

                        // get index
                        auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        if (index >= num_attributes) {
                            // index too big for specified number of features
                            throw invalid_file_format_exception{ fmt::format("Trying to add feature/label at index {} but the maximum index is {}!", index, num_attributes - 1) };
                        }
                        pos = next_pos + 1;

                        // get position of next value
                        next_pos = line.find_first_of(",}", pos);

                        // write parsed value depending on the index
                        if (has_label && index == label_idx) {
                            // write label value
                            is_class_set = true;
                            if constexpr (std::is_same_v<label_type, bool>) {
                                // the std::vector<bool> template specialization is per C++ standard NOT thread safe
                                #pragma omp critical
                                label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                            } else {
                                label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                            }
                        } else {
                            // write feature value has a whitespace!
                            // if the feature index is larger than the label index, the index must be reduced in order to write the feature to the correct data index
                            if (has_label && index > label_idx) {
                                --index;
                            }
                            data(i, index) = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        }

                        // remove already processes part of the line
                        line.remove_prefix(next_pos + 1);
                        line = detail::trim_left(line);
                        pos = 0;
                    }
                    // there should be a class label but none has been found
                    if (has_label && !is_class_set) {
                        throw invalid_file_format_exception{ fmt::format("Missing label for data point \"{}\"!", reader.line(i + num_header_lines)) };
                    }
                } else {
                    // check if the last character is a closing brace
                    if (detail::ends_with(line, '}')) {
                        // no dense line given but a sparse line with a missing opening brace
                        throw invalid_file_format_exception{ fmt::format("Missing opening '{{' for sparse data point \"{}\" description!", line) };
                    }
                    // dense line given
                    const std::vector<std::string_view> line_split = detail::split(line, ',');
                    if (line_split.size() != num_attributes) {
                        throw invalid_file_format_exception{ fmt::format("Invalid number of features and labels! Found {} but should be {}!", line_split.size(), num_attributes) };
                    }
                    for (std::size_t j = 0; j < num_attributes; ++j) {
                        if (has_label && label_idx == j) {
                            // found a label

                            if constexpr (std::is_same_v<label_type, bool>) {
                                // the std::vector<bool> template specialization is per C++ standard NOT thread safe
                                #pragma omp critical
                                label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line_split[j]);
                            } else {
                                label[i] = detail::convert_to<label_type, invalid_file_format_exception>(line_split[j]);
                            }
                        } else {
                            // found data point
                            data(i, j) = detail::convert_to<real_type, invalid_file_format_exception>(line_split[j]);
                        }
                    }
                }

                // check if the parsed label is one of the labels specified in the ARFF file header
                if (has_label && !detail::contains(unique_label, static_cast<label_type>(label[i]))) {
                    throw invalid_file_format_exception{ fmt::format("Found the label \"{}\" which was not specified in the header ({{{}}})!", static_cast<label_type>(label[i]), fmt::join(unique_label, ", ")) };
                }
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

    return std::make_tuple(num_data_points, num_features, std::move(data), has_label ? std::move(label) : std::vector<label_type>{});
}

/**
 * @brief Write the provided @p data and @p labels to the ARFF file @p filename.
 * @details An example file can look like
 * @code
 * @RELATION name
 *
 * @ATTRIBUTE feature_0   numeric
 * @ATTRIBUTE feature_1   numeric
 * @ATTRIBUTE feature_2   numeric
 * @ATTRIBUTE feature_3   numeric
 * @ATTRIBUTE class       {-1,1}
 *
 * @DATA
 * -1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1
 * -0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026,1
 * 0.0,0.60276937379453293,-0.13086851759108944,0.0,-1
 * 0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387,-1
 * 1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514,-1
 * @endcode
 * Note that the output will always be dense, i.e., all features with a value of `0.0` are explicitly written in the resulting file.
 * @tparam label_type the type of the labels (any arithmetic type or std::string)
 * @tparam has_label if `true` the provided labels are also written to the file, if `false` **no** labels are outputted
 * @param[in] filename the filename to write the data to
 * @param[in] data the data points to write to the file
 * @param[in] label the labels to write to the file
 * @note The resulting order of the data points in the ARFF file is unspecified!
 * @note The features are written using zero-based indices!
 */
template <typename label_type, bool has_label>
inline void write_arff_data_impl(const std::string &filename, const soa_matrix<real_type> &data, const std::vector<label_type> &label) {
    if constexpr (has_label) {
        PLSSVM_ASSERT(data.empty() || !label.empty(), "has_label is 'true' but no labels were provided!");
        PLSSVM_ASSERT(data.num_rows() == label.size(), "Number of data points ({}) and number of labels ({}) mismatch!", data.num_rows(), label.size());
    } else {
        PLSSVM_ASSERT(label.empty(), "has_label is 'false' but labels were provided!");
    }

    // create file
    fmt::ostream out = fmt::output_file(filename);
    // write arff header with current time stamp
    out.print("% This data set has been created at {}\n", detail::current_date_time());

    const std::size_t num_data_points = data.num_rows();
    if (num_data_points == 0) {
        // nothing to output
        return;
    }
    const std::size_t num_features = data.num_cols();
    out.print("% {}x{}\n", num_data_points, num_features);

    out.print("@RELATION data_set\n");
    // write arff header for features
    for (std::size_t i = 0; i < num_features; ++i) {
        out.print("@ATTRIBUTE feature_{} NUMERIC\n", i);
    }
    // write arff header for the label if existing
    if constexpr (has_label) {
        const std::set<label_type> available_labels{ label.begin(), label.end() };
        out.print("@ATTRIBUTE class {{{}}}\n", fmt::join(available_labels, ","));
    }
    out.print("@DATA\n");

    // write arff data
    #pragma omp parallel default(none) shared(out, data, label) firstprivate(num_data_points, num_features)
    {
        // all support vectors
        std::string out_string;
        #pragma omp for schedule(dynamic) nowait
        for (std::size_t i = 0; i < num_data_points; ++i) {
            // output data points
            for (std::size_t j = 0; j < num_features - 1; ++j) {
                out_string.append(fmt::format("{:.10e},", data(i, j)));
            }
            out_string.append(fmt::format("{:.10e}", data(i, num_features - 1)));
            // output label if provided
            if constexpr (has_label) {
                out_string.append(fmt::format(",{}", label[i]));
            }
            // output newline at the end
            out_string.push_back('\n');
        }

        #pragma omp critical
        out.print("{}", out_string);
    }
}

/**
 * @brief Write the provided @p data and @p labels to the ARFF file @p filename.
 * @details An example file can look like
 * @code
 * @RELATION name
 *
 * @ATTRIBUTE feature_0   numeric
 * @ATTRIBUTE feature_1   numeric
 * @ATTRIBUTE feature_2   numeric
 * @ATTRIBUTE feature_3   numeric
 * @ATTRIBUTE class       {-1,1}
 *
 * @DATA
 * -1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1
 * -0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026,1
 * 0.0,0.60276937379453293,-0.13086851759108944,0.0,-1
 * 0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387,-1
 * 1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514,-1
 * @endcode
 * Note that the output will always be dense, i.e., all features with a value of `0.0` are explicitly written in the resulting file.
 * @tparam label_type the type of the labels (any arithmetic type or std::string)
 * @param[in] filename the filename to write the data to
 * @param[in] data the data points to write to the file
 * @param[in] label the labels to write to the file
 * @note The resulting order of the data points in the ARFF file is unspecified!
 * @note The features are written using zero-based indices!
 */
template <typename label_type>
inline void write_arff_data(const std::string &filename, const soa_matrix<real_type> &data, const std::vector<label_type> &label) {
    write_arff_data_impl<label_type, true>(filename, data, label);
}

/**
 * @brief Write the provided @p data to the ARFF file @p filename.
 * @details An example file can look like
 * @code
 * @RELATION name
 *
 * @ATTRIBUTE feature_0   numeric
 * @ATTRIBUTE feature_1   numeric
 * @ATTRIBUTE feature_2   numeric
 * @ATTRIBUTE feature_3   numeric
 *
 * @DATA
 * -1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288
 * -0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026
 * 0.0,0.60276937379453293,-0.13086851759108944,0.0
 * 0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387
 * 1.88494043717792,1.00518564317278263,0.298499933047586044,1.6464627048813514
 * @endcode
 * Note that the output will always be dense, i.e., all features with a value of `0.0` are explicitly written in the resulting file.
 * @param[in] filename the filename to write the data to
 * @param[in] data the data points to write to the file
 * @note The resulting order of the data points in the ARFF file is unspecified!
 * @note The features are written using zero-based indices!
 */
inline void write_arff_data(const std::string &filename, const soa_matrix<real_type> &data) {
    write_arff_data_impl<real_type, false>(filename, data, {});
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_ARFF_PARSING_HPP_