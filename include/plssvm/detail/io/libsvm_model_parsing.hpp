#pragma once

#include "plssvm/data_set.hpp"   // plssvm::data_set
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "fmt/compile.h" // FMT_COMPILE
#include "fmt/format.h"  // fmt::format, fmt::format_to
#include "fmt/os.h"      // fmt::ostream
#ifdef _OPENMP
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>    // std::min
#include <cstddef>      // std::size_t
#include <numeric>      // std::accumulate
#include <sstream>      // std::stringstream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move, std::pair
#include <vector>       // std::vector

namespace plssvm::detail::io {

template <typename real_type, typename label_type, typename size_type>
inline std::pair<std::size_t, std::vector<label_type>> read_libsvm_model_header(file_reader &reader, parameter<real_type> &params, real_type &rho, size_type &num_support_vectors) {
    // read libsvm header
    // helper variables
    bool rho_set{ false };
    std::vector<label_type> labels{};
    std::vector<size_type> num_support_vectors_per_class{};

    // parse libsvm model file header
    std::size_t header = 0;
    {
        for (; header < reader.num_lines(); ++header) {
            std::string line{ detail::trim(reader.line(header)) };
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
            } else if (detail::starts_with(line, "kernel_type")) {
                // parse kernel_type, must be linear, polynomial or rbf
                std::istringstream iss{ std::string{ value } };
                iss >> params.kernel;
                if (iss.fail()) {
                    throw invalid_file_format_exception{ fmt::format("Unrecognized kernel type '{}'!", value) };
                }
            } else if (detail::starts_with(line, "gamma")) {
                // parse gamma
                params.gamma = detail::convert_to<decltype(params.gamma)>(value);
            } else if (detail::starts_with(line, "degree")) {
                // parse degree
                params.degree = detail::convert_to<decltype(params.degree)>(value);
            } else if (detail::starts_with(line, "coef0")) {
                // parse coef0
                params.coef0 = detail::convert_to<decltype(params.coef0)>(value);
            } else if (detail::starts_with(line, "nr_class")) {
                // number of classes must be 2
                const auto nr_class = detail::convert_to<unsigned long long>(value);
                if (nr_class != 2) {
                    throw invalid_file_format_exception{ fmt::format("Currently only binary classification is supported, but {} different label where given!", nr_class) };
                }
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_support_vectors = detail::convert_to<size_type>(value);
                if (num_support_vectors == 0) {
                    throw invalid_file_format_exception{ fmt::format("The number of support vectors must be greater than 0, but is {}!", num_support_vectors) };
                }
            } else if (detail::starts_with(line, "rho")) {
                // parse rho, required
                rho = detail::convert_to<real_type>(value);
                rho_set = true;
            } else if (detail::starts_with(line, "label")) {
                // parse label
                labels = detail::split_as<label_type>(value, ' ');

                if (labels.size() < 2) {
                    throw invalid_file_format_exception{ fmt::format("At least two labels must be set, but only {} label was given!", labels.size()) };
                } else if (labels.size() > 2) {
                    throw invalid_file_format_exception{ fmt::format("Currently only binary classification is supported, but {} label where given!", labels.size()) };
                }
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse number of support vectors per class
                num_support_vectors_per_class = detail::split_as<size_type>(value, ' ');

                const auto nr_sv_sum = std::accumulate(num_support_vectors_per_class.begin(), num_support_vectors_per_class.end(), size_type{ 0 });

                if (num_support_vectors_per_class.size() < 2) {
                    throw invalid_file_format_exception{ fmt::format("At least two nr_sv must be set, but only {} was given!", num_support_vectors_per_class.size()) };
                } else if (num_support_vectors_per_class.size() > 2) {
                    throw invalid_file_format_exception{ fmt::format("Currently only binary classification is supported, but {} nr_sv where given!", num_support_vectors_per_class.size()) };
                } else if (nr_sv_sum != num_support_vectors) {
                    throw invalid_file_format_exception{ fmt::format("The total number of support vectors is {}, but the sum of nr_sv is {}!", num_support_vectors, nr_sv_sum) };
                }
            } else if (line == "sv") {
                // start parsing support vectors, required
                break;
            } else {
                throw invalid_file_format_exception{ fmt::format("Unrecognized header entry '{}'! Maybe SV is missing?", reader.line(header)) };
            }
        }
    }

    // additional sanity checks
    if (num_support_vectors == 0) {
        // no total number of support vectors given
        throw invalid_file_format_exception{ "Missing total number of support vectors!" };
    } else if (labels.empty()) {
        // no labels given
        throw invalid_file_format_exception{ "Missing labels!" };
    } else if (num_support_vectors_per_class.empty()) {
        // no count for support vectors per class given
        throw invalid_file_format_exception{ "Missing number of support vectors per class!" };
    } else if (!rho_set) {
        // no rho set
        throw invalid_file_format_exception{ "Missing rho value!" };
    } else if (header + 1 >= reader.num_lines()) {
        // no support vectors given
        throw invalid_file_format_exception{ "Can't parse file: no support vectors are given or SV is missing!" };
    }

    // set label according to model file definition
    std::vector<label_type> data_labels(num_support_vectors);
    std::size_t pos = 0;
    for (size_type i = 0; i < labels.size(); ++i) {
        std::fill(data_labels.begin() + pos, data_labels.begin() + pos + num_support_vectors_per_class[i], labels[i]);
        pos += num_support_vectors_per_class[i];
    }

    return std::make_pair(header, std::move(data_labels));
}

template <typename real_type, typename label_type>
inline std::vector<label_type> write_libsvm_model_header(fmt::ostream &out, const parameter<real_type> &params, const real_type rho, const data_set<real_type, label_type> &data) {
    // save model file header
    out.print("svm_type c_svc\nkernel_type {}\n", params.kernel);
    switch (params.kernel) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            out.print("degree {}\ngamma {}\ncoef0 {}\n", params.degree, params.gamma, params.coef0);
            break;
        case kernel_type::rbf:
            out.print("gamma {}\n", params.gamma);
            break;
    }

    // get the original labels (not the mapped once)
    std::vector<label_type> label_values;
    label_values.reserve(data.num_labels());
    for (const auto& [key, val] : data.mapping()) {
        label_values.push_back(val);
    }
    // count the occurrence of each label
    std::map<label_type, std::size_t> label_counts_map;
    for (const label_type &l : data.labels().value().get()) {
        ++label_counts_map[l];
    }
    // fill vector with number of occurrences in correct order
    std::vector<std::size_t> label_counts(data.num_labels());
    for (typename data_set<real_type, label_type>::size_type i = 0; i < data.num_labels(); ++i) {
        label_counts[i] = label_counts_map[label_values[i]];
    }

    out.print("nr_class {}\nlabel {}\ntotal_sv {}\nnr_sv {}\nrho {}\nSV\n",
              data.num_labels(), fmt::join(label_values, " "), data.num_data_points(), fmt::join(label_counts, " "), rho);

    return label_values;
}

template <typename real_type, typename label_type>
inline void write_libsvm_model_data(fmt::ostream& out, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, const std::vector<label_type> &labels, const std::vector<label_type> &label_order) {
    // format one output-line
    auto format_libsvm_line = [](std::string &output, const real_type a, const std::vector<real_type> &d) {
        static constexpr std::size_t BLOCK_SIZE = 64;
        static constexpr std::size_t CHARS_PER_BLOCK = 128;
        static constexpr std::size_t BUFFER_SIZE = BLOCK_SIZE * CHARS_PER_BLOCK;
        static char buffer[BUFFER_SIZE];
        #pragma omp threadprivate(buffer)

        output.append(fmt::format(FMT_COMPILE("{} "), a));
        for (typename std::vector<real_type>::size_type j = 0; j < d.size(); j += BLOCK_SIZE) {
            char *ptr = buffer;
            for (std::size_t i = 0; i < std::min<std::size_t>(BLOCK_SIZE, d.size() - j); ++i) {
                if (d[j + i] != real_type{ 0.0 }) {
                    ptr = fmt::format_to(ptr, FMT_COMPILE("{}:{:e} "), j + i, d[j + i]);
                }
            }
            output.append(buffer, ptr - buffer);
        }
        output.push_back('\n');
    };
    // TODO: PERFORMANCE!
    std::vector<std::vector<std::string>> out_strings;

    #pragma omp parallel default(none) shared(out, support_vectors, alpha, labels, label_order, format_libsvm_line, out_strings)
    {
        #pragma omp single
        out_strings.resize(omp_get_num_threads(), std::vector<std::string>(label_order.size()));

        // format all lines
        #pragma omp for
        for (typename std::vector<real_type>::size_type i = 0; i < alpha.size(); ++i) {
            format_libsvm_line(out_strings[omp_get_thread_num()][std::find(label_order.begin(), label_order.end(), labels[i]) - label_order.begin()], alpha[i], support_vectors[i]);
        }
    }

    // output strings
    for (typename std::vector<label_type>::size_type i = 0; i < label_order.size(); ++i) {
        for (typename std::vector<std::vector<std::string>>::size_type j = 0; j < out_strings.size(); ++j) {
            out.print("{}", out_strings[j][i]);
        }
    }
}

}