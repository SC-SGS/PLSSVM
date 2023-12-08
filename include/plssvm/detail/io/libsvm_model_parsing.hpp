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

#include "plssvm/classification_types.hpp"      // plssvm::classification_type
#include "plssvm/constants.hpp"                 // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"                  // plssvm::data_set
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/io/libsvm_parsing.hpp"  // plssvm::detail::io::parse_libsvm_num_features
#include "plssvm/detail/logging.hpp"            // plssvm::detail::log
#include "plssvm/detail/utility.hpp"            // plssvm::detail::current_date_time
#include "plssvm/matrix.hpp"                    // plssvm::soa_matrix
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/verbosity_levels.hpp"          // plssvm::verbosity_level

#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format, fmt::format_to
#include "fmt/os.h"       // fmt::ostream, fmt::output_file
#ifdef _OPENMP
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>    // std::swap, std::is_sorted, std::adjacent_find, std::find_first_of, std::lower_bound, std::min, std::fill, std::all_of
#include <array>        // std::array
#include <cstddef>      // std::size_t
#include <exception>    // std::exception_ptr, std::exception, std::current_exception, std::rethrow_exception
#include <iterator>     // std::distance
#include <limits>       // std::numeric_limits::max
#include <map>          // std::map
#include <memory>       // std::unique_ptr, std::make_unique
#include <numeric>      // std::accumulate
#include <set>          // std::set
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Calculates the one-dimensional index in the alpha vector given two classes for the binary one vs. one classifier.
 * @details The binary classifiers for four classes are: 0v1, 0vs2, 0vs3, 1vs2, 1vs3, and 2vs3.
 *          Example function calls with the respective results are: `x_vs_y_to_idx(1, 2, 4) == 3` or `x_vs_y_to_idx(3, 1, 4) == 4`.
 * @note If `x > y` swaps the to values since, e.g., `3vs2` isn't defined and will be mapped to `2vs3`.
 * @param[in] x the first class of the binary classifier
 * @param[in] y the second class if the binary classifier
 * @param[in] num_classes the number of different classes
 * @return the one-dimensional index of the classification pair (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::size_t x_vs_y_to_idx(std::size_t x, std::size_t y, const std::size_t num_classes) {
    PLSSVM_ASSERT(x != y, "Can't compute the index for the binary classification of {}vs{}!", x, y);
    PLSSVM_ASSERT(num_classes > 1, "There must be at least two classes!");
    PLSSVM_ASSERT(x < num_classes, "The class x ({}) must be smaller than the total number of classes ({})!", x, num_classes);
    PLSSVM_ASSERT(y < num_classes, "The class y ({}) must be smaller than the total number of classes ({})!", y, num_classes);

    // e.g., 3vs2 isn't defined -> map it to 2vs3
    if (x > y) {
        std::swap(x, y);
    }
    const std::size_t idx = (num_classes * (num_classes - 1) / 2) - (num_classes - x) * ((num_classes - x) - 1) / 2 + y - x - 1;

    PLSSVM_ASSERT(idx < num_classes * (num_classes - 1) / 2,
                  "The final index ({}) must be smaller than the total number of binary classifiers ({}) for {} different classes!",
                  idx,
                  num_classes * (num_classes - 1) / 2,
                  num_classes);
    return idx;
}

/**
 * @brief Calculates the index in the alpha vector given the index set of the two classes @p i and @p j in the current binary classifier.
 * @details As an example, if the index sets are defined by [0, 2, 4] and [6, 8, 10] and the @p idx_to_find is `4`, returns `2`, if @p idx_to find is `10`, returns `5`.
 * @param[in] i the first class of the binary classifier
 * @param[in] j the second class of the binary classifier
 * @param[in] index_sets the whole indices, used to (implicitly) create the index set of the current binary classifier
 * @param[in] idx_to_find to index of the support vector to find the correct alpha index in the current binary classifier
 * @return the alpha index (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::size_t calculate_alpha_idx(std::size_t i, std::size_t j, const std::vector<std::vector<std::size_t>> &index_sets, const std::size_t idx_to_find) {
    PLSSVM_ASSERT(i != j, "Can't compute the index for {} == {}!", i, j);
    PLSSVM_ASSERT(index_sets.size() > 1, "At least two index sets must be provided!");
    PLSSVM_ASSERT(i < index_sets.size(), "The index i ({}) must be smaller than the total number of indices ({})!", i, index_sets.size());
    PLSSVM_ASSERT(j < index_sets.size(), "The index j ({}) must be smaller than the total number of indices ({})!", j, index_sets.size());
    PLSSVM_ASSERT(std::is_sorted(index_sets[i].cbegin(), index_sets[i].cend()) && std::is_sorted(index_sets[j].cbegin(), index_sets[j].cend()), "The index sets must be sorted in ascending order!");
    PLSSVM_ASSERT(std::adjacent_find(index_sets[i].cbegin(), index_sets[i].cend()) == index_sets[i].cend() && std::adjacent_find(index_sets[j].cbegin(), index_sets[j].cend()) == index_sets[j].cend(), "All indices in one index set must be unique!");
    PLSSVM_ASSERT(std::find_first_of(index_sets[i].cbegin(), index_sets[i].cend(), index_sets[j].cbegin(), index_sets[j].cend()) == index_sets[i].cend(), "The content of both index sets must be disjoint!");

    // the order is predefined -> switch order in order to return the correct index
    if (i > j) {
        std::swap(i, j);
    }

    std::size_t global_idx{ 0 };
    const auto i_it = std::lower_bound(index_sets[i].cbegin(), index_sets[i].cend(), idx_to_find);
    if (i_it != index_sets[i].cend() && *i_it == idx_to_find) {
        // index found
        global_idx = std::distance(index_sets[i].cbegin(), i_it);
    } else {
        // index not yet found
        global_idx = index_sets[i].size() + std::distance(index_sets[j].cbegin(), std::lower_bound(index_sets[j].cbegin(), index_sets[j].cend(), idx_to_find));
    }

    PLSSVM_ASSERT(global_idx < index_sets[i].size() + index_sets[j].size(), "The global index ({}) for the provided index to find ({}) must be smaller than the combined size of both index sets ({} + {})!",
                  global_idx, idx_to_find, index_sets[i].size(), index_sets[j].size());

    return global_idx;
}

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
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
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
 * @throws plssvm::invalid_file_format_exception if the number of provided labels is not the same as the value of 'nr_class'
 * @throws plssvm::invalid_file_format_exception if the number of support vectors per class ('nr_sv') is missing
 * @throws plssvm::invalid_file_format_exception if the number of provided number of support vectors per class is not the same as the value of 'nr_class'
 * @throws plssvm::invalid_file_format_exception if the number of sum of all number of support vectors per class is not the same as the value of 'total_sv'
 * @throws plssvm::invalid_file_format_exception if no support vectors have been provided in the data section
 * @attention The PLSSVM model file is only compatible with LIBSVM for the one vs. one classification type.
 * @return the necessary header information: [the SVM parameter; the values of rho; the labels; the different classes; the number of support vectors per class; num_header_lines] (`[[nodiscard]]`)
 */
template <typename label_type>
[[nodiscard]] inline std::tuple<plssvm::parameter, std::vector<real_type>, std::vector<label_type>, std::vector<label_type>, std::vector<std::size_t>, std::size_t> parse_libsvm_model_header(const std::vector<std::string_view> &lines) {
    // data to read
    plssvm::parameter params{};
    std::vector<real_type> rho{};
    std::size_t num_support_vectors{};

    // helper variables
    bool svm_type_set{ false };
    bool kernel_type_set{ false };
    bool nr_class_set{ false };
    bool total_sv_set{ false };
    bool rho_set{ false };
    bool label_set{ false };
    bool nr_sv_set{ false };
    std::size_t nr_class{};
    std::vector<label_type> labels{};
    std::vector<std::size_t> num_support_vectors_per_class{};

    // parse libsvm model file header
    std::size_t header_line = 0;
    {
        for (; header_line < lines.size(); ++header_line) {
            // get the current line and convert it to lower case
            std::string line{ detail::trim(lines[header_line]) };
            detail::to_lower_case(line);

            // separate value from model header entry
            std::string_view value{ line };
            value.remove_prefix(std::min(value.find_first_of(' '), value.size()));
            value = detail::trim(value);

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
                // number of classes must be greater or equal than two
                nr_class = detail::convert_to<unsigned long long>(value);
                // read the number of classes (number of different labels)
                nr_class_set = true;
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_support_vectors = detail::convert_to<std::size_t>(value);
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
                const std::set<label_type> unique_labels(labels.cbegin(), labels.cend());
                if (labels.size() != unique_labels.size()) {
                    throw invalid_file_format_exception{ fmt::format("Provided {} labels but only {} of them was/where unique!", labels.size(), unique_labels.size()) };
                }
                // read the labels
                label_set = true;
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse number of support vectors per class
                num_support_vectors_per_class = detail::split_as<std::size_t>(value, ' ');
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
    // calculate the number of support as sum of the support vectors per class
    const auto nr_sv_sum = std::accumulate(num_support_vectors_per_class.begin(), num_support_vectors_per_class.end(), std::size_t{ 0 });
    if (nr_sv_sum != num_support_vectors) {
        throw invalid_file_format_exception{ fmt::format("The total number of support vectors is {}, but the sum of nr_sv is {}!", num_support_vectors, nr_sv_sum) };
    }
    // check if no support vectors are given
    if (header_line + 1 >= lines.size()) {
        throw invalid_file_format_exception{ "Can't parse file: no support vectors are given or SV is missing!" };
    }
    // check for the minimum number of required rho values
    if (rho.size() < std::min(nr_class, nr_class * (nr_class - 1) / 2)) {
        throw invalid_file_format_exception{ fmt::format("Provided {} rho values but at least min({}, {}) = {} are needed!",
                                                         rho.size(),
                                                         nr_class,
                                                         nr_class * (nr_class - 1) / 2,
                                                         std::min(nr_class, nr_class * (nr_class - 1) / 2)) };
    }

    // set label according to model file definition
    std::vector<label_type> data_labels(num_support_vectors);
    std::size_t pos = 0;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        std::fill(data_labels.begin() + pos, data_labels.begin() + pos + num_support_vectors_per_class[i], labels[i]);
        pos += num_support_vectors_per_class[i];
    }

    return std::make_tuple(params, rho, std::move(data_labels), std::move(labels), std::move(num_support_vectors_per_class), header_line + 1);
}

/**
 * @brief Parse all data points and weights (alpha values) using the file @p reader, ignoring all empty lines and lines starting with an `#`.
 * @details An example data section of a file with three classes and one vs. all classification can look like
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
 * @param[in] reader the file_reader used to read the LIBSVM data
 * @param[in] num_sv_per_class the number of support vectors per class
 * @param[in] skipped_lines the number of lines that should be skipped at the beginning
 * @note The features must be provided with one-based indices!
 * @throws plssvm::invalid_file_format_exception if no features could be found (may indicate an empty file)
 * @throws plssvm::invalid_file_format_exception if the provided total @p num_sv_per_class is greater or less than the number of read support vectors
 * @throws plssvm::invalid_file_format_exception if more or less weights than @p num_classes could be found
 * @throws plssvm::invalid_file_format_exception if a weight couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if a feature index couldn't be converted to `unsigned long`
 * @throws plssvm::invalid_file_format_exception if a feature value couldn't be converted to the provided @p real_type
 * @throws plssvm::invalid_file_format_exception if the provided LIBSVM file uses zero-based indexing (LIBSVM mandates one-based indices)
 * @throws plssvm::invalid_file_format_exception if the feature (indices) are not given in a strictly increasing order
 * @attention The PLSSVM model file is only compatible with LIBSVM for the one vs. one classification type.
 * @return a std::tuple containing: [num_data_points, num_features, data_points, labels, classification type] (`[[nodiscard]]`)
 */
[[nodiscard]] inline std::tuple<std::size_t, std::size_t, soa_matrix<real_type>, std::vector<aos_matrix<real_type>>, classification_type> parse_libsvm_model_data(const file_reader &reader, const std::vector<std::size_t> &num_sv_per_class, const std::size_t skipped_lines) {
    PLSSVM_ASSERT(reader.is_open(), "The file_reader is currently not associated with a file!");
    PLSSVM_ASSERT(num_sv_per_class.size() > 1, "At least two classes must be present!");
    PLSSVM_ASSERT(skipped_lines <= reader.num_lines(), "Tried to skipp {} lines, but only {} are present!", skipped_lines, reader.num_lines());

    // parse sizes
    const std::size_t num_data_points = reader.num_lines() - skipped_lines;
    const std::size_t num_features = parse_libsvm_num_features(reader.lines(), skipped_lines);

    // no features were parsed -> invalid file
    if (num_features == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file: no data points are given!") };
    }
    // mismatching number of data points and num_sv_per_class
    if (std::reduce(num_sv_per_class.cbegin(), num_sv_per_class.cend()) != num_data_points) {
        throw invalid_file_format_exception{ fmt::format("Found {} support vectors, but it should be {}!", num_data_points, std::reduce(num_sv_per_class.cbegin(), num_sv_per_class.cend())) };
    }

    // create vector containing the data and label
    soa_matrix<real_type> data{ num_data_points, num_features, PADDING_SIZE, PADDING_SIZE };
    const std::size_t max_num_alpha_values = num_sv_per_class.size();  // OAA needs more alpha values than OAO
    aos_matrix<real_type> alpha{ max_num_alpha_values, num_data_points, PADDING_SIZE, PADDING_SIZE };
    bool is_oaa{ false };
    bool is_oao{ false };

    std::exception_ptr parallel_exception;

    #pragma omp parallel default(none) shared(reader, skipped_lines, data, alpha, parallel_exception, is_oaa, is_oao) firstprivate(num_data_points, max_num_alpha_values)
    {
        #pragma omp for
        for (std::size_t i = 0; i < num_data_points; ++i) {
            try {
                const std::string_view line = reader.line(skipped_lines + i);
                unsigned long last_index = 0;

                // parse the alpha (weight) values
                std::string_view::size_type pos = 0;
                const std::string_view::size_type first_colon = line.find_first_of(":\n");
                std::size_t alpha_val{ 0 };
                while (true) {
                    const std::string_view::size_type next_pos = line.find_first_of(" \n", pos);
                    if (first_colon >= next_pos) {
                        if (alpha_val >= max_num_alpha_values) {
                            throw invalid_file_format_exception{ fmt::format("Can't parse file: needed at most {} alpha values, but more ({}) were provided!", max_num_alpha_values, alpha_val + 1) };
                        }

                        // get alpha value
                        alpha(alpha_val, i) = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos));
                        pos = next_pos + 1;
                        ++alpha_val;
                    } else {
                        break;
                    }
                }
                if (alpha_val < max_num_alpha_values - 1) {
                    throw invalid_file_format_exception{ fmt::format("Can't parse file: needed at least {} alpha values, but fewer ({}) were provided!", max_num_alpha_values - 1, alpha_val) };
                }

                // check whether we read a file given OAA classification or OAO classification
                if (alpha_val == max_num_alpha_values) {
                    #pragma omp critical
                    is_oaa = true;
                } else if (alpha_val == max_num_alpha_values - 1) {
                    #pragma omp critical
                    is_oao = true;
                }

                // get data
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
                    data(i, index) = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
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
            }
        }
    }

    // rethrow if an exception occurred inside the parallel region
    if (parallel_exception) {
        std::rethrow_exception(parallel_exception);
    }

    classification_type classification{};
    std::vector<aos_matrix<real_type>> alpha_vec{};
    if (is_oaa && is_oao) {
        // invalid model file
        throw invalid_file_format_exception{ "Can't distinguish between OAA and OAO in the given model file!" };
    } else if (is_oaa) {
        classification = classification_type::oaa;
        alpha_vec = std::vector<aos_matrix<real_type>>{ std::move(alpha) };
    } else if (is_oao) {
        classification = classification_type::oao;
        // last vector entry must be ignored!
        // remap alpha vector from the read one to 01 02 03 12 13 23 etc.
        const std::size_t num_classes = num_sv_per_class.size();

        // resize all alpha vectors to the correct, final size
        alpha_vec.resize(calculate_number_of_classifiers(classification_type::oao, num_classes));
        for (std::size_t i = 0; i < num_classes; ++i) {
            for (std::size_t j = i + 1; j < num_classes; ++j) {
                alpha_vec[x_vs_y_to_idx(i, j, num_classes)] = aos_matrix<real_type>{ 1, num_sv_per_class[i] + num_sv_per_class[j], plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };
            }
        }
        std::vector<std::size_t> oao_alpha_indices(alpha_vec.size(), 0);

        // loop over all classes
        std::size_t running_idx{ 0 };
        for (std::size_t nr_sv = 0; nr_sv < num_classes; ++nr_sv) {
            // loop over all data points in the specific class
            // note: the data points are sorted according to the labels/classes by definition
            for (std::size_t i = 0; i < num_sv_per_class[nr_sv]; ++i) {
                // running_a index: since in the case nr_sv == a, one index has to be skipped and, therefore, a can't be used directly
                std::size_t running_a{ 0 };
                // loop over all alpha values for the current data point
                // note: OAO saves one alpha value less than there are classes
                for (std::size_t a = 0; a < num_classes - 1; ++a) {
                    // x vs x isn't defined
                    if (a == nr_sv) {
                        ++running_a;
                    }
                    // get the current alpha value
                    const real_type alpha_val = alpha(a, running_idx);
                    // calculate to which alpha vector the alpha value should be added to
                    const std::size_t idx = x_vs_y_to_idx(nr_sv, running_a, num_classes);
                    // add the alpha value
                    alpha_vec[idx](0, oao_alpha_indices[idx]++) = alpha_val;
                    ++running_a;
                }
                // update the running index (otherwise a prefix sum over num_sv_per_class is needed)
                ++running_idx;
            }
        }
    }

    return std::make_tuple(num_data_points, num_features, std::move(data), std::move(alpha_vec), classification);
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
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in,out] out the output-stream to write the header information to
 * @param[in] params the SVM parameters
 * @param[in] rho the rho values for the different classes resulting from the hyperplane learning
 * @param[in] data the data used to create the model
 * @attention The PLSSVM model file is only compatible with LIBSVM for the one vs. one classification type.
 * @return the order of the different classes as it should appear in the following data section (`[[nodiscard]]`)
 */
template <typename label_type>
[[nodiscard]] inline std::vector<label_type> write_libsvm_model_header(fmt::ostream &out, const plssvm::parameter &params, const std::vector<real_type> &rho, const data_set<label_type> &data) {
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");
    PLSSVM_ASSERT(!rho.empty(), "At least one rho value must be provided!");

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
    const std::vector<label_type> classes = data.classes().value();

    // count the occurrence of each label
    std::map<label_type, std::size_t> label_counts_map;
    const std::vector<label_type> labels = data.labels().value();
    for (const label_type &l : labels) {
        ++label_counts_map[l];
    }
    // fill vector with number of occurrences in correct order
    std::vector<std::size_t> label_counts(data.num_classes());
    for (typename data_set<label_type>::size_type i = 0; i < data.num_classes(); ++i) {
        label_counts[i] = label_counts_map[classes[i]];
    }

    out_string += fmt::format("nr_class {}\nlabel {}\ntotal_sv {}\nnr_sv {}\nrho {:.10e}\nSV\n",
                              data.num_classes(),
                              fmt::join(classes, " "),
                              data.num_data_points(),
                              fmt::join(label_counts, " "),
                              fmt::join(rho, " "));

    // print model header
    detail::log(verbosity_level::full | verbosity_level::libsvm,
                "\n{}\n",
                out_string);
    // write model header to file
    out.print("{}", out_string);

    return classes;
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
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in] filename the file to write the LIBSVM model to
 * @param[in] params the SVM parameters
 * @param[in] classification the used multi-class classification strategy
 * @param[in] rho the rho value resulting from the hyperplane learning
 * @param[in] alpha the weights learned by the SVM
 * @param[in] index_sets index sets containing the SV indices per class
 * @param[in] data the data used to create the model
 * @attention The PLSSVM model file is only compatible with LIBSVM for the one vs. one classification type.
 */
template <typename label_type>
inline void write_libsvm_model_data(const std::string &filename, const plssvm::parameter &params, const classification_type classification, const std::vector<real_type> &rho, const std::vector<aos_matrix<real_type>> &alpha, const std::vector<std::vector<std::size_t>> &index_sets, const data_set<label_type> &data) {
    PLSSVM_ASSERT(!filename.empty(), "The provided model filename must not be empty!");
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");
    PLSSVM_ASSERT(rho.size() == calculate_number_of_classifiers(classification, data.num_classes()),
                  "The number of rho values is {} but must be {} ({})",
                  rho.size(),
                  calculate_number_of_classifiers(classification, data.num_classes()),
                  classification);
#if defined(PLSSVM_ASSERT_ENABLED)
    switch (classification) {
        case classification_type::oaa:
            // weights
            PLSSVM_ASSERT(alpha.size() == 1, "In case of OAA, the alpha vector may only contain one matrix as entry, but has {}!", alpha.size());
            PLSSVM_ASSERT(alpha.front().num_rows() == calculate_number_of_classifiers(classification, data.num_classes()), "The number of rows in the matrix must be {}, but is {}!", calculate_number_of_classifiers(classification, data.num_classes()), alpha.front().num_rows());
            PLSSVM_ASSERT(alpha.front().num_cols() == data.num_data_points(), "The number of weights ({}) must be equal to the number of support vectors ({})!", alpha.front().num_cols(), data.num_data_points());

            // indices: NO index sets are calculated for OAA
            PLSSVM_ASSERT(index_sets.empty(), "There shouldn't be any index sets for the OAA classification, but {} were found!", index_sets.size());
            break;
        case classification_type::oao:
            // weights
            PLSSVM_ASSERT(alpha.size() == calculate_number_of_classifiers(classification, data.num_classes()), "The number of matrices in the alpha vector must contain {} entries, but contains {} entries!", calculate_number_of_classifiers(classification, data.num_classes()), alpha.size());
            PLSSVM_ASSERT(std::all_of(alpha.cbegin(), alpha.cend(), [](const aos_matrix<real_type> &matr) { return matr.num_rows() == 1; }), "In case of OAO, each matrix may only contain one row!");

            // indices: only calculated for OAO
            PLSSVM_ASSERT(index_sets.size() == data.num_classes(), "The number of index sets ({}) must be equal to the number of different classes ({})!", index_sets.size(), data.num_classes());
            PLSSVM_ASSERT(std::accumulate(index_sets.cbegin(), index_sets.cend(), std::size_t{ 0 }, [](const std::size_t count, const std::vector<std::size_t> &set) { return count + set.size(); }) == data.num_data_points(), "Each data point must have exactly one entry in the index set!");
            PLSSVM_ASSERT(std::all_of(index_sets.cbegin(), index_sets.cend(), [](const std::vector<std::size_t> &set) { return std::is_sorted(set.cbegin(), set.cend()); }), "All index sets must be sorted in ascending order!");
            PLSSVM_ASSERT(std::all_of(index_sets.cbegin(), index_sets.cend(), [](const std::vector<std::size_t> &set) { return std::adjacent_find(set.cbegin(), set.cend()) == set.cend(); }), "All indices in one index set must be unique!");

            // note: computationally expensive!!!
            for (std::size_t i = 0; i < index_sets.size(); ++i) {
                for (std::size_t j = i + 1; j < index_sets.size(); ++j) {
                    PLSSVM_ASSERT(std::find_first_of(index_sets[i].cbegin(), index_sets[i].cend(), index_sets[j].cbegin(), index_sets[j].cend()) == index_sets[i].cend(), "All index sets must be pairwise unique, but index sets {} and {} share at least one index!", i, j);
                }
            }
            break;
    }
#endif

    const soa_matrix<real_type> &support_vectors = data.data();
    const std::vector<label_type> &labels = data.labels().value();
    const std::size_t num_features = data.num_features();
    const std::size_t num_classes = data.num_classes();
    const std::size_t num_alpha_per_point = classification == classification_type::oaa ? num_classes : num_classes - 1;

    // create file
    fmt::ostream out = fmt::output_file(filename);
    // write timestamp for current date time
    // note: commented out since the resulting model file cannot be read be LIBSVM
    //    out.print("# This model file has been created at {}\n", detail::current_date_time());

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
    constexpr std::size_t STRING_BUFFER_SIZE = std::size_t{ 1024 } * std::size_t{ 1024 };

    // format one output-line
    auto format_libsvm_line = [](std::string &output, const std::vector<real_type> &a, const soa_matrix<real_type> &d, const std::size_t point) {
        static constexpr std::size_t STACK_BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static std::array<char, STACK_BUFFER_SIZE> buffer{};
        #pragma omp threadprivate(buffer)

        output.append(fmt::format("{:.10e} ", fmt::join(a, " ")));
        for (typename std::vector<real_type>::size_type j = 0; j < d.num_cols(); j += BLOCK_SIZE) {
            char *ptr = buffer.data();
            for (std::size_t i = 0; i < std::min<std::size_t>(BLOCK_SIZE, d.num_cols() - j); ++i) {
                if (d(point, j + i) != real_type{ 0.0 }) {
                    // add 1 to the index since LIBSVM assumes 1-based feature indexing
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:.10e} "), j + i + 1, d(point, j + i));
                }
            }
            output.append(buffer.data(), ptr - buffer.data());
        }
        output.push_back('\n');
    };

    // initialize volatile array
    auto counts = std::make_unique<volatile int[]>(label_order.size() + 1);
    counts[0] = std::numeric_limits<int>::max();
    #pragma omp parallel default(none) shared(counts, alpha, format_libsvm_line, label_order, labels, support_vectors, out, index_sets) firstprivate(num_features, num_classes, num_alpha_per_point, classification)
    {
        // preallocate string buffer, only ONE allocation
        std::string out_string;
        out_string.reserve(STRING_BUFFER_SIZE + (num_features + num_alpha_per_point) * CHARS_PER_BLOCK);  // oversubscribe buffer that at least one additional line fits into it
        std::vector<real_type> alpha_per_point(num_alpha_per_point);

        // loop over all classes, since they must be sorted
        for (typename std::vector<label_type>::size_type l = 0; l < label_order.size(); ++l) {
            // the support vectors with the l-th class
            #pragma omp for nowait
            for (typename std::vector<real_type>::size_type i = 0; i < support_vectors.num_rows(); ++i) {
                if (labels[i] == label_order[l]) {
                    switch (classification) {
                        case classification_type::oaa:
                            for (std::size_t a = 0; a < num_alpha_per_point; ++a) {
                                alpha_per_point[a] = alpha.front()(a, i);
                            }
                            break;
                        case classification_type::oao:
                            for (std::size_t j = 0, pos = 0; j < num_classes; ++j) {
                                if (l != j) {
                                    const std::size_t idx = x_vs_y_to_idx(l, j, num_classes);
                                    const aos_matrix<real_type> &alpha_vec = alpha[idx];
                                    const std::size_t sv_idx = calculate_alpha_idx(l, j, index_sets, i);
                                    alpha_per_point[pos] = alpha_vec(0, sv_idx);
                                    ++pos;
                                }
                            }
                            break;
                    }
                    format_libsvm_line(out_string, alpha_per_point, support_vectors, i);

                    // if the buffer is full, write it to the file
                    if (out_string.size() > STRING_BUFFER_SIZE) {
                        // wait for all threads to write support vectors for previous class
#ifdef _OPENMP
                        while (counts[l] < omp_get_num_threads()) {
                        }
#endif
                        #pragma omp critical
                        {
                            out.print("{}", out_string);
                        }
                        // clear buffer
                        out_string.clear();
                    }
                }
            }
            // wait for all threads to write support vectors for previous class
#ifdef _OPENMP
            while (counts[l] < omp_get_num_threads()) {
            }
#endif

            #pragma omp critical
            {
                if (!out_string.empty()) {
                    out.print("{}", out_string);
                    out_string.clear();
                }
                counts[l + 1] = counts[l + 1] + 1;
            }
        }
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_LIBSVM_MODEL_PARSING_HPP_