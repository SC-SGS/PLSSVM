/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/CSVM.hpp"

#include "plssvm/detail/file_reader.hpp"     // plssvm::detail::file_reader
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::convert_to, plssvm::detail::starts_with, plssvm::detail::ends_with, plssvm::detail::trim_left
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::format, fmt::print

#if __has_include(<omp.h>)
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>    // std::max, std::transform
#include <cctype>       // std::toupper
#include <chrono>       // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <exception>    // std::exception_ptr, std::exception, std::current_exception, std::rethrow_exception
#include <ostream>      // std::ofstream, std::ios::out, std::ios::trunc
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm {

// read and parse file
template <typename T>
void CSVM<T>::parse_file(const std::string &filename) {
    if (detail::ends_with(filename, ".arff")) {
        parse_arff(filename);
    } else {
        parse_libsvm(filename);
    }
}

// read and parse a libsvm file
template <typename T>
void CSVM<T>::parse_libsvm(const std::string &filename) {
    auto start_time = std::chrono::steady_clock::now();

    detail::file_reader f{ filename, '#' };

    data_.resize(f.num_lines());
    value_.resize(f.num_lines());

    size_type max_size = 0;
    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for reduction(max:max_size)
        for (size_type i = 0; i < data_.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = f.line(i);

                // get class
                size_type pos = line.find_first_of(' ');
                value_[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(0, pos)) > real_type{ 0.0 } ? 1 : -1;
                // value[i] = std::copysign(1.0, detail::convert_to<real_type>(line.substr(0, pos)));

                // get data
                std::vector<real_type> vline(max_size);
                while (true) {
                    size_type next_pos = line.find_first_of(':', pos);
                    // no further data points
                    if (next_pos == std::string_view::npos) {
                        break;
                    }

                    // get index
                    const auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                    if (index >= vline.size()) {
                        vline.resize(index + 1);
                    }
                    pos = next_pos + 1;

                    // get value
                    next_pos = line.find_first_of(' ', pos);
                    vline[index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                    pos = next_pos;
                }
                max_size = std::max(max_size, vline.size());
                data_[i] = std::move(vline);
            } catch (const std::exception &e) {
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

    #pragma omp parallel for
    for (size_type i = 0; i < data_.size(); ++i) {
        data_[i].resize(max_size);
    }

    // update values
    num_data_points_ = data_.size();
    num_features_ = max_size;

    // no features were parsed -> invalid file
    if (num_features_ == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file '{}'!", filename) };
    }

    // update gamma
    if (gamma_ == 0) {
        gamma_ = 1. / num_features_;
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Read {} data points with {} features in {} using the libsvm parser.\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

// read and parse an ARFF file
template <typename T>
void CSVM<T>::parse_arff(const std::string &filename) {
    auto start_time = std::chrono::steady_clock::now();

    detail::file_reader f{ filename, '%' };
    size_type max_size = 0;

    // parse arff header
    size_type header = 0;
    {
        for (; header < f.num_lines(); ++header) {
            std::string line{ f.line(header) };
            std::transform(line.begin(), line.end(), line.begin(), [](const char c) { return std::toupper(c); });
            if (detail::starts_with(line, "@RELATION")) {
                // ignore relation
                continue;
            } else if (detail::starts_with(line, "@ATTRIBUTE")) {
                if (line.find("NUMERIC") == std::string::npos) {
                    throw invalid_file_format_exception{ fmt::format("Can only use NUMERIC features, but '{}' was given!", line) };
                }
                // add a feature
                ++max_size;
            } else if (detail::starts_with(line, "@DATA")) {
                // finished reading header -> start parsing data
                break;
            }
        }
    }

    // something went wrong, i.e. no @ATTRIBUTE fields
    if (max_size == 0) {
        throw invalid_file_format_exception{ "Invalid file format!" };
    }

    data_.resize(f.num_lines() - (header + 1));
    value_.resize(f.num_lines() - (header + 1));

    #pragma omp parallel for
    for (size_type i = 0; i < data_.size(); ++i) {
        data_[i].resize(max_size - 1);
    }

    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for
        for (size_type i = 0; i < data_.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = f.line(i + header + 1);
                //
                if (detail::starts_with(line, '@')) {
                    // read @ inside data section
                    throw invalid_file_format_exception{ fmt::format("Read @ inside data section!: {}", line) };
                }

                // parse sparse or dense data point definition
                if (detail::starts_with(line, '{')) {
                    // missing closing }
                    if (!detail::ends_with(line, '}')) {
                        throw invalid_file_format_exception{ "Missing closing '}' for sparse data point description!" };
                    }
                    // sparse line
                    bool is_class_set = false;
                    size_type pos = 1;
                    while (true) {
                        size_type next_pos = line.find_first_of(' ', pos);
                        // no further data points
                        if (next_pos == std::string_view::npos) {
                            break;
                        }

                        // get index
                        const auto index = detail::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;

                        // get value
                        next_pos = line.find_first_of(",}", pos);

                        // write parsed value depending on the index
                        if (index == max_size - 1) {
                            is_class_set = true;
                            value_[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos)) > real_type{ 0.0 } ? 1 : -1;
                        } else {
                            data_[i][index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        }

                        // remove already processes part of the line
                        line.remove_prefix(next_pos + 1);
                        line = detail::trim_left(line);
                        pos = 0;
                    }
                    // no class label found
                    if (!is_class_set) {
                        throw invalid_file_format_exception{ "Missing class for data point!" };
                    }
                } else {
                    // dense line
                    size_type pos = 0;
                    for (size_type j = 0; j < max_size - 1; ++j) {
                        size_type next_pos = line.find_first_of(',', pos);
                        if (next_pos == std::string_view::npos) {
                            throw invalid_file_format_exception{ fmt::format("Invalid number of features! Found {} but should be {}.", j, max_size - 1) };
                        }
                        data_[i][j] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;
                    }
                    value_[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos)) > real_type{ 0.0 } ? 1 : -1;
                }
            } catch (const std::exception &e) {
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

    // update values
    num_data_points_ = data_.size();
    num_features_ = max_size - 1;

    // update gamma
    if (gamma_ == 0) {
        gamma_ = 1. / num_features_;
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Read {} data points with {} features in {} using the arff parser.\n",
                   num_data_points_,
                   num_features_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

template <typename T>
void CSVM<T>::write_model(const std::string &model_name) {
    auto start_time = std::chrono::steady_clock::now();

    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (size_type i = 0; i < alpha_.size(); ++i) {
        if (value_[i] > 0) {
            ++count_pos;
        }
        if (value_[i] < 0) {
            ++count_neg;
        }
        if (alpha_[i] == cost_) {
            ++nBSV;
        }
    }

    // create libsvm model header
    std::string libsvm_model_header = fmt::format(
        "svm_type c_svc\n"
        "kernel_type {}\n"
        "nr_class 2\n"
        "total_sv {}\n"
        "rho {}\n"
        "label 1 -1\n"
        "nr_sv {} {}\n"
        "SV\n",
        kernel_,
        count_pos + count_neg,
        -bias_,
        count_pos,
        count_neg);

    // terminal output
    if (print_info_) {
        fmt::print("\nOptimization finished\n{}\n", libsvm_model_header);
    }

    // create model file
    std::ofstream model{ model_name.data(), std::ios::out | std::ios::trunc };
    model << libsvm_model_header;

    // format one output-line
    auto format_libsmv_line = [](real_type a, const std::vector<real_type> &d) -> std::string {
        std::string line;
        line += fmt::format("{} ", a);
        for (size_type j = 0; j < d.size(); ++j) {
            if (d[j] != 0.0) {
                line += fmt::format("{}:{:e} ", j, d[j]);
            }
        }
        line.push_back('\n');
        return line;
    };

    volatile int count = 0;
    #pragma omp parallel
    {
        // all support vectors with class 1
        std::string out_pos;
        #pragma omp for nowait
        for (size_type i = 0; i < alpha_.size(); ++i) {
            if (value_[i] > 0) {
                out_pos += format_libsmv_line(alpha_[i], data_[i]);
            }
        }

        #pragma omp critical
        {
            model.write(out_pos.data(), static_cast<std::streamsize>(out_pos.size()));
            count++;
            #pragma omp flush(count, model)
        }

        // all support vectors with class -1
        std::string out_neg;
        #pragma omp for nowait
        for (size_type i = 0; i < alpha_.size(); ++i) {
            if (value_[i] < 0) {
                out_neg += format_libsmv_line(alpha_[i], data_[i]);
            }
        }

        // wait for all threads to write support vectors for class 1
#if __has_include(<omp.h>)
        while (count < omp_get_num_threads()) {
        }
#else
        #pragma omp barrier
#endif

        #pragma omp critical
        model.write(out_neg.data(), static_cast<std::streamsize>(out_neg.size()));
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Wrote model file with {} support vectors in {}.\n",
                   count_pos + count_neg,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

// explicitly instantiate template class
template class CSVM<float>;
template class CSVM<double>;

}  // namespace plssvm
