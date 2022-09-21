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

#include "plssvm/data_set.hpp"       // plssvm::data_set
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/parameter.hpp"      // plssvm::parameter

#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/format.h"   // fmt::format, fmt::format_to
#include "fmt/os.h"       // fmt::ostream, fmt::output_file
#ifdef _OPENMP
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>    // std::min, std::fill
#include <cstddef>      // std::size_t
#include <iostream>     // std::cout
#include <map>          // std::map
#include <memory>       // std::unique_ptr
#include <numeric>      // std::accumulate
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <tuple>        // std::tuple, std::make_tuple
#include <utility>      // std::move, std::pair
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief Parse the LIBSVM model file header.
 * @details An example LIBSVM model file header for the linear kernel and two labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * label 1 -1
 * total_sv 5
 * nr_sv 2 3
 * SV
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @tparam size_type the size type
 * @param[in] lines the LIBSVM model file header to parse>
 * @throws plssvm::invalid_file_format_exception if an invalid 'svm_type' has been provided, i.e., 'svm_type' is not 'c_csc'
 * @throws plssvm::invalid_file_format_exception if an invalid 'kernel_type has been provided
 * @throws plssvm::invalid_file_format_exception if the number of support vectors ('total_sv') is zero
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
 * @throws plssvm::invalid_file_format_exception if the number of labels is not two
 * @return the necessary header information: [the SVM parameter, the value of rho, the labels of the data points, num_header_lines] (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type, typename size_type>
[[nodiscard]] inline std::tuple<parameter<real_type>, real_type, std::vector<label_type>, std::size_t> parse_libsvm_model_header(const std::vector<std::string_view> &lines) {
    // data to read
    plssvm::parameter<real_type> params;
    real_type rho = 0.0;
    size_type num_support_vectors;

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
                iss >> params.kernel;
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
                rho = detail::convert_to<real_type>(value);
                // read the rho value
                rho_set = true;
            } else if (detail::starts_with(line, "label")) {
                // parse available label, note: we can't use value here since we want to preserve the case of the labels
                std::string_view original_line = detail::trim(lines[header_line]);
                original_line.remove_prefix(std::min(original_line.find_first_of(' ') + 1, original_line.size()));
                original_line = detail::trim_left(original_line);
                labels = detail::split_as<label_type>(original_line, ' ');
                if (labels.size() < 2) {
                    throw invalid_file_format_exception{ fmt::format("At least two labels must be set, but only {} label ([{}]) was given!", labels.size(), fmt::join(labels, ", ")) };
                }
                // read the labels
                label_set = true;
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse number of support vectors per class
                num_support_vectors_per_class = detail::split_as<size_type>(value, ' ');
                if (num_support_vectors_per_class.size() < 2) {
                    throw invalid_file_format_exception{ fmt::format("At least two nr_sv must be set, but only {} ([{}]) was given!", num_support_vectors_per_class.size(), fmt::join(num_support_vectors_per_class, ", ")) };
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
    switch (params.kernel) {
        case plssvm::kernel_type::linear:
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
        case plssvm::kernel_type::polynomial:
            break;
        case plssvm::kernel_type::rbf:

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
        throw invalid_file_format_exception{ "Missing rho value!" };
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

    // current limitation
    if (nr_class != 2) {
        throw invalid_file_format_exception{ fmt::format("Currently only binary classification is supported, but {} different label where given!", nr_class) };
    }

    return std::make_tuple(params, rho, std::move(data_labels), header_line + 1);
}

/**
 * @brief Write the LIBSVM model file header to @p out.
 * @details An example LIBSVM model file header for the linear kernel and two labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * label 1 -1
 * total_sv 5
 * nr_sv 2 3
 * SV
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in,out] out the output-stream to write the header information to
 * @param[in] params the SVM parameters
 * @param[in] rho the rho value resulting from the hyperplane learning
 * @param[in] data the data used to create the model
 * @return the order of the labels; necessary to write the data points in the correct order (`[[nodiscard]]`)
 */
template <typename real_type, typename label_type>
[[nodiscard]] inline std::vector<label_type> write_libsvm_model_header(fmt::ostream &out, const parameter<real_type> &params, const real_type rho, const data_set<real_type, label_type> &data) {
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");

    // save model file header
    std::string out_string = fmt::format("svm_type c_svc\nkernel_type {}\n", params.kernel);
    // save the SVM parameter information based on the used kernel_type
    switch (params.kernel) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            out_string += fmt::format("degree {}\ngamma {}\ncoef0 {}\n", params.degree, params.gamma, params.coef0);
            break;
        case kernel_type::rbf:
            out_string += fmt::format("gamma {}\n", params.gamma);
            break;
    }

    // get the original labels (not the mapped once)
    const typename plssvm::data_set<real_type, label_type>::label_mapper mapper = data.mapping().value();
    const std::vector<label_type> label_values = mapper.labels();

    // count the occurrence of each label
    std::map<label_type, std::size_t> label_counts_map;
    const std::vector<label_type> labels = data.labels().value();
    for (const label_type &l : labels) {
        ++label_counts_map[l];
    }
    // fill vector with number of occurrences in correct order
    std::vector<std::size_t> label_counts(data.num_labels());
    for (typename data_set<real_type, label_type>::size_type i = 0; i < data.num_labels(); ++i) {
        label_counts[i] = label_counts_map[label_values[i]];
    }

    out_string += fmt::format("nr_class {}\nlabel {}\ntotal_sv {}\nnr_sv {}\nrho {}\nSV\n",
                              data.num_labels(),
                              fmt::join(label_values, " "),
                              data.num_data_points(),
                              fmt::join(label_counts, " "),
                              rho);

    // print model header
    std::cout << '\n'
              << out_string << '\n';
    // write model header to file
    out.print(out_string);

    return label_values;
}

/**
 * @brief Write the LIBSVM model to the file @p filename.
 * @details An example LIBSVM model file for the linear kernel and two labels could look like
 * @code
 * svm_type c_svc
 * kernel_type linear
 * nr_class 2
 * rho 0.37330625882191915
 * label 1 -1
 * total_sv 5
 * nr_sv 2 3
 * SV
 * -0.17609610490769723 1:-1.117828e+00 2:-2.908719e+00 3:6.663834e-01 4:1.097883e+00
 * 0.8838187731213127 1:-5.282118e-01 2:-3.358810e-01 3:5.168730e-01 4:5.460446e-01
 * -0.47971257671001616 1:-2.098121e-01 2:6.027694e-01 3:-1.308685e-01 4:1.080525e-01
 * 0.0034556484621847128 1:1.884940e+00 2:1.005186e+00 3:2.984999e-01 4:1.646463e+00
 * -0.23146573996578407 1:5.765022e-01 2:1.014056e+00 3:1.300943e-01 4:7.261914e-01
 * @endcode
 * @tparam real_type the floating point type
 * @tparam label_type the type of the labels (any arithmetic type, except bool, or std::string)
 * @param[in] filename the file to write the LIBSVM model to
 * @param[in] params the SVM parameters
 * @param[in] rho the rho value resulting from the hyperplane learning
 * @param[in] alpha the weights learned by the SVM
 * @param[in] data the data used to create the model
 */
template <typename real_type, typename label_type>
inline void write_libsvm_model_data(const std::string &filename, const parameter<real_type> &params, const real_type rho, const std::vector<real_type> &alpha, const data_set<real_type, label_type> &data) {
    PLSSVM_ASSERT(data.has_labels(), "Cannot write a model file that does not include labels!");
    PLSSVM_ASSERT(alpha.size() == data.num_data_points(), "The number of weights ({}) doesn't match the number of data points ({})!", alpha.size(), data.num_data_points());

    const std::vector<std::vector<real_type>> &support_vectors = data.data();
    const std::vector<label_type> &labels = data.labels().value();
    const std::size_t num_features = data.num_features();

    // create file
    fmt::ostream out = fmt::output_file(filename);

    // write header information
    const std::vector<label_type> label_order = write_libsvm_model_header(out, params, rho, data);

    // the maximum size of one formatted LIBSVM entry, e.g., 1234:1.365363e+10
    // biggest number representable as std::size_t: 18446744073709551615 -> 20 chars
    // scientific notation: 3 chars (number in front of decimal separator including a sign + decimal separator) + 10 chars (part after the decimal separator, specified during formatting) +
    //                      5 chars exponent (e + sign + maximum potential exponent (308 -> 3 digits)
    // separators: 2 chars (: between index and feature + whitespace after feature value)
    // -> 40 chars in total
    // -> increased to 48 chars to be on the safe side
    constexpr std::size_t CHARS_PER_BLOCK = 48;
    // results in 48 B * 128 B = 6 KiB stack buffer per thread
    constexpr std::size_t BLOCK_SIZE = 128;
    // use 1 MiB as buffer per thread
    constexpr std::size_t STRING_BUFFER_SIZE = 1024 * 1024;

    // format one output-line
    auto format_libsvm_line = [=](std::string &output, const real_type a, const std::vector<real_type> &d) {
        static constexpr std::size_t STACK_BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static char buffer[STACK_BUFFER_SIZE];
        #pragma omp threadprivate(buffer)

        output.append(fmt::format(FMT_COMPILE("{:.10e} "), a));
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

        // support vectors with the first class
        #pragma omp for nowait
        for (typename std::vector<real_type>::size_type i = 0; i < alpha.size(); ++i) {
            if (labels[i] == label_order[0]) {
                format_libsvm_line(out_string, alpha[i], support_vectors[i]);

                // if the buffer is full, write it to the file
                if (out_string.size() > STRING_BUFFER_SIZE) {
                    #pragma omp critical
                    {
                        out.print(out_string);
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
                out.print(out_string);
                out_string.clear();
            }
            counts[0]++;
            #pragma omp flush(counts, out)
        }

        for (typename std::vector<label_type>::size_type l = 1; l < label_order.size(); ++l) {
            // the support vectors with the i-th class

            #pragma omp for nowait
            for (typename std::vector<real_type>::size_type i = 0; i < alpha.size(); ++i) {
                if (labels[i] == label_order[l]) {
                    format_libsvm_line(out_string, alpha[i], support_vectors[i]);

                    // if the buffer is full, write it to the file
                    if (out_string.size() > STRING_BUFFER_SIZE) {
                        #pragma omp critical
                        {
                            out.print(out_string);
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
                    out.print(out_string);
                    out_string.clear();
                }
                counts[l]++;
                #pragma omp flush(counts, out)
            }
        }
    }
}

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_LIBSVM_MODEL_PARSING_HPP_