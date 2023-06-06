/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements parsing functions for the LIBSVM model file.
 */

#ifndef PLSSVM_DETAIL_IO_LIBSVM_MODEL_PARSING_HPP_
#define PLSSVM_DETAIL_IO_LIBSVM_MODEL_PARSING_HPP_
#pragma once

#include "plssvm/data_set.hpp"                  // plssvm::data_set
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/io/libsvm_parsing.hpp"  // plssvm::detail::io::parse_libsvm_num_features
#include "plssvm/detail/logger.hpp"             // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/utility.hpp"            // plssvm::detail::current_date_time
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format, fmt::format_to
#include "fmt/os.h"       // fmt::ostream, fmt::output_file
#ifdef _OPENMP
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>    // std::min, std::fill
#include <cstddef>      // std::size_t
#include <map>          // std::map
#include <memory>       // std::unique_ptr
#include <numeric>      // std::accumulate
#include <set>          // std::set
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move, std::pair
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Parse the modified LIBSVM model file header.
 * @details An example modified LIBSVM model file header for the linear kernel and three labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 3
 * rho 0.37330625882191915 0.19351329465391632 1.98151327406835528
 * label cat dog mouse
 * total_sv 8
 * nr_sv 4 2 2
 * SV
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @tparam size_type the size type
 * @param[in] lines the LIBSVM model file header to parse>
 * @throws plssvm::invalid_file_format_exception if an invalid 'svm_type' has been provided, i.e., 'svm_type' is not 'c_csc'
 * @throws plssvm::invalid_file_format_exception if an invalid 'kernel_type has been provided
 * @throws plssvm::invalid_file_format_exception if the number of support vectors ('total_sv') is zero
 * @throws plssvm::invalid_file_format_exception if less than two rho values have been provided
 * @throws plssvm::invalid_file_format_exception if less than two labels have been provided
 * @throws plssvm::invalid_file_format_exception if less than two number of support vectors per label have been provided
 * @throws plssvm::invalid_file_format_exception if an invalid header entry has been read
 * @throws plssvm::invalid_file_format_exception if the 'svm_type' is missing
 * @throws plssvm::invalid_file_format_exception if the 'kernel_type' is missing
 * @throws plssvm::invalid_file_format_exception if SVM parameter are explicitly provided that are not used in the give kernel (e.g., 'gamma' is provided for the 'linear' kernel)
 * @throws plssvm::invalid_file_format_exception if the number of classes ('nr_class') is missing
 * @throws plssvm::invalid_file_format_exception if the total number of support vectors ('total_sv') is missing
 * @throws plssvm::invalid_file_format_exception if the value for rho is missing
 * @throws plssvm::invalid_file_format_exception if the labels are missing
 * @throws plssvm::invalid_file_format_exception if the number of provided rho values is not the same as the value of 'nr_class' or 1 for binary classification
 * @throws plssvm::invalid_file_format_exception if the number of provided labels is not the same as the value of 'nr_class'
 * @throws plssvm::invalid_file_format_exception if the number of support vectors per class ('nr_sv') is missing
 * @throws plssvm::invalid_file_format_exception if the number of provided number of support vectors per class is not the same as the value of 'nr_class'
 * @throws plssvm::invalid_file_format_exception if the number of sum of all number of support vectors per class is not the same as the value of 'total_sv'
 * @throws plssvm::invalid_file_format_exception if no support vectors have been provided in the data section
 * @throws plssvm::invalid_file_format_exception if the number of labels is not two
 * @attention Due to using one vs. all (OAA) for multi-class classification, the model file isn't LIBSVM conform except for binary classification!
 * @return the necessary header information: [the SVM parameter, the values of rho, the labels, the different classes, num_header_lines] (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type, typename size_type>
[[nodiscard]] inline std::tuple<plssvm::parameter, std::vector<real_type>, std::vector<label_type>, std::vector<label_type>, std::size_t> parse_libsvm_model_header(const std::vector<std::string_view> &lines) {
    // data to read
    plssvm::parameter params{};
    std::vector<real_type> rho{};
    size_type num_support_vectors{};

    // helper variables
    bool svm_type_set{ false };
    bool kernel_type_set{ false };
    bool nr_class_set{ false };
    bool total_sv_set{ false };
    bool rho_set{ false };
    bool label_set{ false };
    bool nr_sv_set{ false };
    size_type nr_class{};
    std::vector<label_type> labels{};
    std::vector<size_type> num_support_vectors_per_class{};

    // parse libsvm model file header
    std::size_t header_line = 0;
    {
        for (; header_line < lines.size(); ++header_line) {
            // get the current line and convert it to lower case
            std::string line{ detail::trim(lines[header_line]) };
            detail::to_lower_case(line);

            // separate value from model header entry
            std::string_view value{ line };
            value.remove_prefix(std::min(value.find_first_of(' ') + 1, value.size()));
            value = detail::trim_left(value);

            if (detail::starts_with(line, "svm_type")) {
                // svm_type must be c_svc
                if (value != "c_svc") {
                    throw invalid_file_format_exception{ fmt::format("Can only use c_svc as svm_type, but '{}' was given!", value) };
                }
                // read the svm_type
                svm_type_set = true;
            } else if (detail::starts_with(line, "kernel_type")) {
                // parse kernel_type, must be linear, polynomial or rbf
                std::istringstream iss{ std::string{ value } };
                iss >> params.kernel_type;
                if (iss.fail()) {
                    throw invalid_file_format_exception{ fmt::format("Unrecognized kernel type '{}'!", value) };
                }
                // read the kernel_type
                kernel_type_set = true;
            } else if (detail::starts_with(line, "gamma")) {
                // parse gamma
                params.gamma = detail::convert_to<typename decltype(params.gamma)::value_type>(value);
            } else if (detail::starts_with(line, "degree")) {
                // parse degree
                params.degree = detail::convert_to<typename decltype(params.degree)::value_type>(value);
            } else if (detail::starts_with(line, "coef0")) {
                // parse coef0
                params.coef0 = detail::convert_to<typename decltype(params.coef0)::value_type>(value);
            } else if (detail::starts_with(line, "nr_class")) {
                // number of classes must be 2
                nr_class = detail::convert_to<unsigned long long>(value);
                // read the number of classes (number of different labels)
                nr_class_set = true;
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_support_vectors = detail::convert_to<size_type>(value);
                if (num_support_vectors == 0) {
                    throw invalid_file_format_exception{ "The number of support vectors must be greater than 0!" };
                }
                // read the number of support vectors
                total_sv_set = true;
            } else if (detail::starts_with(line, "rho")) {
                // parse rho, required
                rho = detail::split_as<real_type>(value, ' ');
                if (rho.empty()) {
                    throw invalid_file_format_exception{ "At least one rho value must be set, but none was given!" };
                }
                // read the rho value
                rho_set = true;
            } else if (detail::starts_with(line, "label")) {
                // parse available label, note: we can't use value here since we want to preserve the case of the labels
                std::string_view original_line = detail::trim(lines[header_line]);
                original_line.remove_prefix(std::min(original_line.find_first_of(' ') + 1, original_line.size()));
                original_line = detail::trim_left(original_line);
                labels = detail::split_as<label_type>(original_line, ' ');
                if (labels.size() < 2) {
                    throw invalid_file_format_exception{ "At least two labels must be set, but only one label was given!" };
                }
                // check if all labels are unique
                std::set<label_type> unique_labels{};
                for (const label_type &label : labels) {
                    unique_labels.insert(label);
                }
                if (labels.size() != unique_labels.size()) {
                    throw invalid_file_format_exception{ fmt::format("Provided {} labels but only {} of them was/where unique!", labels.size(), unique_labels.size()) };
                }
                // read the labels
                label_set = true;
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse number of support vectors per class
                num_support_vectors_per_class = detail::split_as<size_type>(value, ' ');
                if (num_support_vectors_per_class.size() < 2) {
                    throw invalid_file_format_exception{ "At least two nr_sv must be set, but only one was given!" };
                }
                // read the number of support vectors per class
                nr_sv_set = true;
            } else if (line == "sv") {
                // start parsing support vectors, required
                break;
            } else {
                throw invalid_file_format_exception{ fmt::format("Unrecognized header entry '{}'! Maybe SV is missing?", lines[header_line]) };
            }
        }
    }

    // additional sanity checks
    if (!svm_type_set) {
        throw invalid_file_format_exception{ "Missing svm_type!" };
    }
    if (!kernel_type_set) {
        throw invalid_file_format_exception{ "Missing kernel_type!" };
    }
    // check provided values based on kernel_type
    switch (params.kernel_type) {
        case plssvm::kernel_function_type::linear:
            if (!params.degree.is_default()) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the degree parameter which is not used in the linear kernel!" };
            }
            if (!params.gamma.is_default()) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the gamma parameter which is not used in the linear kernel!" };
            }
            if (!params.coef0.is_default()) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the coef0 parameter which is not used in the linear kernel!" };
            }
            break;
        case plssvm::kernel_function_type::polynomial:
            break;
        case plssvm::kernel_function_type::rbf:

            if (!params.degree.is_default()) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the degree parameter which is not used in the radial basis function kernel!" };
            }
            if (!params.coef0.is_default()) {
                throw invalid_file_format_exception{ "Explicitly provided a value for the coef0 parameter which is not used in the radial basis function kernel!" };
            }
            break;
    }
    if (!nr_class_set) {
        throw invalid_file_format_exception{ "Missing number of different classes nr_class!" };
    }
    if (!total_sv_set) {
        throw invalid_file_format_exception{ "Missing total number of support vectors total_sv!" };
    }
    if (!rho_set) {
        throw invalid_file_format_exception{ "Missing rho values!" };
    }
    if (!label_set) {
        throw invalid_file_format_exception{ "Missing class label specification!" };
    }
    // number of different labels must match the number of classes
    if (nr_class != labels.size()) {
        throw invalid_file_format_exception{ fmt::format("The number of classes (nr_class) is {}, but the provided number of different labels is {} (label)!", nr_class, labels.size()) };
    }
    if (!nr_sv_set) {
        throw invalid_file_format_exception{ "Missing number of support vectors per class nr_sv!" };
    }
    // number of different label numbers must match the number of classes
    if (nr_class != num_support_vectors_per_class.size()) {
        throw invalid_file_format_exception{ fmt::format("The number of classes (nr_class) is {}, but the provided number of different labels is {} (nr_sv)!", nr_class, num_support_vectors_per_class.size()) };
    }
    // the number of rho values must match the number of classes
    if (nr_class == 2) {
        if (rho.size() != 1) {
            throw invalid_file_format_exception{ fmt::format("The number of rho values (rho) is {}, but must be 1 for binary classification!", rho.size()) };
        }
    } else {
        if (rho.size() != nr_class) {
            throw invalid_file_format_exception{ fmt::format("The number of rho values (rho) is {}, but the provided number of different labels is {} (nr_class)!", rho.size(), num_support_vectors_per_class.size()) };
        }
    }
    // calculate the number of support as sum of the support vectors per class
    const auto nr_sv_sum = std::accumulate(num_support_vectors_per_class.begin(), num_support_vectors_per_class.end(), size_type{ 0 });
    if (nr_sv_sum != num_support_vectors) {
        throw invalid_file_format_exception{ fmt::format("The total number of support vectors is {}, but the sum of nr_sv is {}!", num_support_vectors, nr_sv_sum) };
    }
    // check if no support vectors are given
    if (header_line + 1 >= lines.size()) {
        throw invalid_file_format_exception{ "Can't parse file: no support vectors are given or SV is missing!" };
    }

    // set label according to model file definition
    std::vector<label_type> data_labels(num_support_vectors);
    std::size_t pos = 0;
    for (size_type i = 0; i < labels.size(); ++i) {
        std::fill(data_labels.begin() + pos, data_labels.begin() + pos + num_support_vectors_per_class[i], labels[i]);
        pos += num_support_vectors_per_class[i];
    }

    return std::make_tuple(params, rho, std::move(data_labels), std::move(labels), header_line + 1);
}

/**
 * @brief Parse all data points and weights (alpha values) using the file @p reader, ignoring all empty lines and lines starting with an `#`.
 * @details An example data section of a file with three classes can look like
 * @code
 * 5.3748085208e-01 -3.3759056567e-01 -3.0674790664e-01 1:5.6909150126e-01 2:1.7385902768e-01 3:-1.2544805985e-01 4:-2.9571509449e-01
 * -8.4228254963e-01 6.0157355473e-01 5.3640238234e-01 1:9.5925668223e-01 2:-7.6818755962e-01 3:6.3621573833e-01 4:1.6085979487e-01
 * -7.0935195569e-01 3.9655742831e-01 7.6465777835e-01 1:5.4545142940e-02 2:2.8991780132e-01 3:7.2598021499e-01 4:-2.3469246049e-01
 * 3.8588219373e-01 -3.8733973551e-01 -5.6883369521e-01 1:3.6828886105e-01 2:8.0120546896e-01 3:6.1204205296e-01 4:-1.2074044818e-02
 * 5.5593569052e-01 -7.5317247576e-01 -3.0608401400e-01 1:-1.0777233473e-02 2:-2.0706684519e-01 3:6.6972496646e-01 4:-1.7887705784e-01
 * -2.2729386622e-01 -3.5287900824e-01 -4.1344671298e-01 1:2.9114472700e-01 2:3.1334616548e-02 3:5.5877309773e-01 4:2.4442300699e-01
 * 1.2048548972e+00 -4.5199224962e-02 -9.1864283628e-01 1:5.6054664148e-02 2:6.6037150647e-01 3:2.6221749918e-01 4:3.7216083001e-02
 * -6.9331414578e-01 4.0173372861e-01 8.7072111162e-01 1:2.3912649397e-02 2:3.4604367826e-01 3:-2.5696199624e-01 4:6.0591633490e-01
 * @endcode
 * @tparam real_type the floating point type
 * @param[in] reader the file_reader used to read the LIBSVM data
 * @param[in] num_alpha_values the number of different labels in the data set
 * @param[in] skipped_lines the number of lines that should be skipped at the beginning
 * @note The features must be provided with one-based indices!
 * @throws plssvm::invalid_file_format_exception if no features could be found (may indicate an empty file)
 * @throws plssvm::invalid_file_format_exception if more or less weights than @p num_different_labels could be found
 * @throws plssvm::invalid_file_format_exception if a weight couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if a feature index couldn't be converted to `unsigned long`
 * @throws plssvm::invalid_file_format_exception if a feature value couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if the provided LIBSVM file uses zero-based indexing (LIBSVM mandates one-based indices)
 * @throws plssvm::invalid_file_format_exception if the feature (indices) are not given in a strictly increasing order
 * @attention Due to using one vs. all (OAA) for multi-class classification, the model file isn't LIBSVM conform except for binary classification!
 * @return a std::tuple containing: [num_data_points, num_features, data_points, labels] (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, std::vector<std::vector<real_type>>, std::vector<std::vector<real_type>>> parse_libsvm_model_data(const file_reader &reader, const std::size_t num_alpha_values, const std::size_t skipped_lines) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");
    PLSSVM_ASSERT(num_alpha_values >= 1, "At least one alpha value must be present!");
    PLSSVM_ASSERT(num_alpha_values != 2, "Two alpha values may never be present (binary classification as special case only uses 1 alpha value)!");
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
    std::vector<std::vector<real_type>> alpha(num_alpha_values, std::vector<real_type>(num_data_points));

    std::exception_ptr parallel_exception;

    #pragma omp parallel default(none) shared(std::cerr, reader, skipped_lines, data, alpha, parallel_exception) firstprivate(num_features, num_alpha_values)
    {
        #pragma omp for
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size(); ++i) {
            try {
                std::string_view line = reader.line(skipped_lines + i);
                unsigned long last_index = 0;

                // parse the alpha (weight) values
                std::string_view::size_type pos = 0;
                const std::string_view::size_type first_colon = line.find_first_of(":\n");
                for (std::size_t a = 0; a < num_alpha_values; ++a) {
                    const std::string_view::size_type next_pos = line.find_first_of(" \n", pos);
                    if (first_colon >= next_pos) {
                        // get alpha value
                        alpha[a][i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos));
                        pos = next_pos + 1;
                    } else {
                        throw invalid_file_format_exception{ fmt::format("Can't parse file: need {} alpha values, but only {} were given!", num_alpha_values, a) };
                    }
                }
                // check whether too many alpha values are provided
                if (line.find_first_of(" \n", pos) < first_colon && line.find_first_of(" \n", pos) != std::string_view::npos) {
                    throw invalid_file_format_exception{ fmt::format("Can't parse file: too many alpha values were given!") };
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
            }
        }
    }

    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }

    return std::make_tuple(num_data_points, num_features, std::move(data), std::move(alpha));
}

/**
 * @brief Write the modified LIBSVM model file header to @p out.
 * @details An example modified LIBSVM model file header for the linear kernel and three labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 3
 * rho 0.37330625882191915 0.19351329465391632 1.98151327406835528
 * label cat dog mouse
 * total_sv 8
 * nr_sv 4 2 2
 * SV
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in,out] out the output-stream to write the header information to
 * @param[in] params the SVM parameters
 * @param[in] rho the rho values for the different classes resulting from the hyperplane learning
 * @param[in] data the data used to create the model
 * @attention Due to using one vs. all (OAA) for multi-class classification, the model file isn't LIBSVM conform except for binary classification!
 */
template <typename real_type, typename label_type>
inline std::vector<label_type> write_libsvm_model_header(fmt::ostream &out, const plssvm::parameter &params, const std::vector<real_type> &rho, const data_set<real_type, label_type> &data) {
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");
    PLSSVM_ASSERT(data.num_different_labels() == 2 ? rho.size() == 1 : rho.size() == data.num_different_labels(),
                  "The number of rho values ({}) must be equal to the number of different labels ({}) or must be 1 for binary classification!", rho.size(), data.num_different_labels());

    // save model file header
    std::string out_string = fmt::format("svm_type c_svc\nkernel_type {}\n", params.kernel_type);
    // save the SVM parameter information based on the used kernel_type
    switch (params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            out_string += fmt::format("degree {}\ngamma {}\ncoef0 {}\n", params.degree, params.gamma, params.coef0);
            break;
        case kernel_function_type::rbf:
            out_string += fmt::format("gamma {}\n", params.gamma);
            break;
    }

    // get the original labels (not the mapped once)
    const std::vector<label_type> label_values = data.different_labels().value();

    // count the occurrence of each label
    std::map<label_type, std::size_t> label_counts_map;
    const std::vector<label_type> labels = data.labels().value();
    for (const label_type &l : labels) {
        ++label_counts_map[l];
    }
    // fill vector with number of occurrences in correct order
    std::vector<std::size_t> label_counts(data.num_different_labels());
    for (typename data_set<real_type, label_type>::size_type i = 0; i < data.num_different_labels(); ++i) {
        label_counts[i] = label_counts_map[label_values[i]];
    }

    out_string += fmt::format("nr_class {}\nlabel {}\ntotal_sv {}\nnr_sv {}\nrho {}\nSV\n",
                              data.num_different_labels(),
                              fmt::join(label_values, " "),
                              data.num_data_points(),
                              fmt::join(label_counts, " "),
                              fmt::join(rho, " "));

    // print model header
    detail::log(verbosity_level::full | verbosity_level::libsvm,
                "\n{}\n", out_string);
    // write model header to file
    out.print("{}", out_string);

    return label_values;
}

/**
 * @brief Write the modified LIBSVM model to the file @p filename.
 * @details An example modified LIBSVM model file for the linear kernel and three labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 3
 * rho 0.37330625882191915 0.19351329465391632 1.98151327406835528
 * label cat dog mouse
 * total_sv 8
 * nr_sv 4 2 2
 * SV
 * 5.3748085208e-01 -3.3759056567e-01 -3.0674790664e-01  1:5.6909150126e-01 2:1.7385902768e-01 3:-1.2544805985e-01 4:-2.9571509449e-01
 * -8.4228254963e-01 6.0157355473e-01 5.3640238234e-01 1:9.5925668223e-01 2:-7.6818755962e-01 3:6.3621573833e-01 4:1.6085979487e-01
 * -7.0935195569e-01 3.9655742831e-01 7.6465777835e-01 1:5.4545142940e-02 2:2.8991780132e-01 3:7.2598021499e-01 4:-2.3469246049e-01
 * 3.8588219373e-01 -3.8733973551e-01 -5.6883369521e-01 1:3.6828886105e-01 2:8.0120546896e-01 3:6.1204205296e-01 4:-1.2074044818e-02
 * 5.5593569052e-01 -7.5317247576e-01 -3.0608401400e-01 1:-1.0777233473e-02 2:-2.0706684519e-01 3:6.6972496646e-01 4:-1.7887705784e-01
 * -2.2729386622e-01 -3.5287900824e-01 -4.1344671298e-01 1:2.9114472700e-01 2:3.1334616548e-02 3:5.5877309773e-01 4:2.4442300699e-01
 * 1.2048548972e+00 -4.5199224962e-02 -9.1864283628e-01 1:5.6054664148e-02 2:6.6037150647e-01 3:2.6221749918e-01 4:3.7216083001e-02
 * -6.9331414578e-01 4.0173372861e-01 8.7072111162e-01 1:2.3912649397e-02 2:3.4604367826e-01 3:-2.5696199624e-01 4:6.0591633490e-01
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in] filename the file to write the LIBSVM model to
 * @param[in] params the SVM parameters
 * @param[in] rho the rho value resulting from the hyperplane learning
 * @param[in] alpha the weights learned by the SVM
 * @param[in] data the data used to create the model
 * @attention Due to using one vs. all (OAA) for multi-class classification, the model file isn't LIBSVM conform except for binary classification!
 */
template <typename real_type, typename label_type>
inline void write_libsvm_model_data(const std::string &filename, const plssvm::parameter &params, const std::vector<real_type> &rho, const std::vector<std::vector<real_type>> &alpha, const data_set<real_type, label_type> &data) {
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");
    PLSSVM_ASSERT(data.num_different_labels() == 2 ? rho.size() == 1 : rho.size() == data.num_different_labels(),
                  "The number of rho values ({}) must be equal to the number of different labels ({}) or must be 1 for binary classification!", rho.size(), data.num_different_labels());
    PLSSVM_ASSERT(data.num_different_labels() == 2 ? alpha.size() == 1 : alpha.size() == data.num_different_labels(),
                  "The number of weight vectors ({}) must be equal to the number of different labels ({}) or must be 1 for binary classification!", alpha.size(), data.num_different_labels());
    PLSSVM_ASSERT(std::all_of(alpha.cbegin(), alpha.cend(), [&alpha](const std::vector<real_type> &a) { return a.size() == alpha.front().size(); }), "The number of weights per class must be equal!");
    PLSSVM_ASSERT(alpha.front().size() == data.num_data_points(), "The number of weights ({}) must be equal to the number of support vectors ({})!", alpha.front().size(), data.num_data_points());

    const std::vector<std::vector<real_type>> &support_vectors = data.data();
    const std::vector<label_type> &labels = data.labels().value();
    const std::size_t num_features = data.num_features();

    // create file
    fmt::ostream out = fmt::output_file(filename);
    // write timestamp for current date time
    out.print("# This model file has been created at {}\n", detail::current_date_time());

    // write header information
    const std::vector<label_type> label_order = write_libsvm_model_header(out, params, rho, data);

    // the maximum size of one formatted LIBSVM entry, e.g., 1234:1.365363e+10
    // biggest number representable as std::size_t: 18446744073709551615 -> 20 chars
    // scientific notation: 3 chars (number in front of decimal separator including a sign + decimal separator) + 10 chars (part after the decimal separator, specified during formatting) +
    //                      5 chars exponent (e + sign + maximum potential exponent (308 -> 3 digits)
    // separators: 2 chars (: between index and feature + whitespace after feature value)
    // -> 40 chars in total
    // -> increased to 48 chars to be on the safe side
    static constexpr std::size_t CHARS_PER_BLOCK = 48;
    // results in 48 B * 128 B = 6 KiB stack buffer per thread
    static constexpr std::size_t BLOCK_SIZE = 128;
    // use 1 MiB as buffer per thread
    constexpr std::size_t STRING_BUFFER_SIZE = 1024 * 1024;

    // format one output-line
    auto format_libsvm_line = [](std::string &output, const std::vector<real_type> &a, const std::vector<real_type> &d) {
        static constexpr std::size_t STACK_BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static char buffer[STACK_BUFFER_SIZE];
        #pragma omp threadprivate(buffer)

        output.append(fmt::format("{:.10e} ", fmt::join(a, " ")));
        for (typename std::vector<real_type>::size_type j = 0; j < d.size(); j += BLOCK_SIZE) {
            char *ptr = buffer;
            for (std::size_t i = 0; i < std::min<std::size_t>(BLOCK_SIZE, d.size() - j); ++i) {
                if (d[j + i] != real_type{ 0.0 }) {
                    // add 1 to the index since LIBSVM assumes 1-based feature indexing
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:.10e} "), j + i + 1, d[j + i]);
                }
            }
            output.append(buffer, ptr - buffer);
        }
        output.push_back('\n');
    };

    // initialize volatile array
    auto counts = std::make_unique<volatile int[]>(label_order.size());
    #pragma omp parallel default(none) shared(counts, alpha, format_libsvm_line, label_order, labels, support_vectors, out) firstprivate(BLOCK_SIZE, CHARS_PER_BLOCK, num_features)
    {
        // preallocate string buffer, only ONE allocation
        std::string out_string;
        out_string.reserve(STRING_BUFFER_SIZE + (num_features + 1) * CHARS_PER_BLOCK);
        std::vector<real_type> alpha_per_point(alpha.size());

        // support vectors with the first class
        #pragma omp for nowait
        for (typename std::vector<real_type>::size_type i = 0; i < support_vectors.size(); ++i) {
            if (labels[i] == label_order[0]) {
                for (typename std::vector<std::vector<real_type>>::size_type a = 0; a < alpha.size(); ++a) {
                    alpha_per_point[a] = alpha[a][i];
                }
                format_libsvm_line(out_string, alpha_per_point, support_vectors[i]);

                // if the buffer is full, write it to the file
                if (out_string.size() > STRING_BUFFER_SIZE) {
                    #pragma omp critical
                    {
                        out.print("{}", out_string);
                    #pragma omp flush(out)
                    }
                    // clear buffer
                    out_string.clear();
                }
            }
        }

        #pragma omp critical
        {
            if (!out_string.empty()) {
                out.print("{}", out_string);
                out_string.clear();
            }
            counts[0] = counts[0] + 1;
            #pragma omp flush(counts, out)
        }

        for (typename std::vector<label_type>::size_type l = 1; l < label_order.size(); ++l) {
            // the support vectors with the i-th class

            #pragma omp for nowait
            for (typename std::vector<real_type>::size_type i = 0; i < support_vectors.size(); ++i) {
                if (labels[i] == label_order[l]) {
                    for (typename std::vector<std::vector<real_type>>::size_type a = 0; a < alpha.size(); ++a) {
                        alpha_per_point[a] = alpha[a][i];
                    }
                    format_libsvm_line(out_string, alpha_per_point, support_vectors[i]);

                    // if the buffer is full, write it to the file
                    if (out_string.size() > STRING_BUFFER_SIZE) {
                        #pragma omp critical
                        {
                            out.print("{}", out_string);
                        #pragma omp flush(out)
                        }
                        // clear buffer
                        out_string.clear();
                    }
                }
            }
            // wait for all threads to write support vectors for previous class
#ifdef _OPENMP
            while (counts[l - 1] < omp_get_num_threads()) {
            }
#else
    #pragma omp barrier
#endif

            #pragma omp critical
            {
                if (!out_string.empty()) {
                    out.print("{}", out_string);
                    out_string.clear();
                }
                counts[l] = counts[l] + 1;
                #pragma omp flush(counts, out)
            }
        }
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_LIBSVM_MODEL_PARSING_HPP_