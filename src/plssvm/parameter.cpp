/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/parameter.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/file_reader.hpp"           // plssvm::detail::file_reader
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::convert_to, plssvm::detail::starts_with, plssvm::detail::ends_with, plssvm::detail::trim_left
#include "plssvm/exceptions/exceptions.hpp"        // plssvm::invalid_file_format_exception
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::format, fmt::print

#include <algorithm>    // std::max, std::transform
#include <cctype>       // std::toupper
#include <chrono>       // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <exception>    // std::exception_ptr, std::exception, std::current_exception, std::rethrow_exception
#include <memory>       // std::make_shared
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm {

// read and parse file
template <typename T>
void parameter<T>::parse_file(const std::string &filename) {
    if (detail::ends_with(filename, ".arff")) {
        parse_arff(filename);
    } else {
        parse_libsvm(filename);
    }
}

// read and parse a libsvm file
template <typename T>
void parameter<T>::parse_libsvm(const std::string &filename) {
    auto start_time = std::chrono::steady_clock::now();

    detail::file_reader f{ filename, '#' };

    std::vector<std::vector<real_type>> data(f.num_lines());
    std::vector<real_type> value(f.num_lines());

    size_type max_size = 0;
    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for reduction(max \
                          : max_size)
        for (size_type i = 0; i < data.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = f.line(i);

                // get class
                size_type pos = line.find_first_of(' ');
                value[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(0, pos)) > real_type{ 0.0 } ? 1 : -1;
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
                data[i] = std::move(vline);
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
    for (size_type i = 0; i < data.size(); ++i) {
        data[i].resize(max_size);
    }

    // no features were parsed -> invalid file
    if (max_size == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file '{}'!", filename) };
    }

    // update gamma
    if (gamma == 0.0) {
        gamma = real_type{ 1. } / static_cast<real_type>(max_size);
    }

    // update shared pointer
    data_ptr = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data));
    value_ptr = std::make_shared<const std::vector<real_type>>(std::move(value));

    auto end_time = std::chrono::steady_clock::now();
    if (print_info) {
        fmt::print("Read {} data points with {} features in {} using the libsvm parser.\n",
                   data_ptr->size(),
                   max_size,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

// read and parse an ARFF file
template <typename T>
void parameter<T>::parse_arff(const std::string &filename) {
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

    std::vector<std::vector<real_type>> data(f.num_lines() - (header + 1));
    std::vector<real_type> value(f.num_lines() - (header + 1));

    #pragma omp parallel for
    for (size_type i = 0; i < data.size(); ++i) {
        data[i].resize(max_size - 1);
    }

    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for
        for (size_type i = 0; i < data.size(); ++i) {
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
                            value[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos)) > real_type{ 0.0 } ? 1 : -1;
                        } else {
                            data[i][index] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
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
                        data[i][j] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;
                    }
                    value[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(pos)) > real_type{ 0.0 } ? 1 : -1;
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

    // update gamma
    if (gamma == 0.0) {
        gamma = real_type{ 1. } / static_cast<real_type>(max_size - 1);
    }

    // update shared pointer
    data_ptr = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data));
    value_ptr = std::make_shared<const std::vector<real_type>>(std::move(value));

    auto end_time = std::chrono::steady_clock::now();
    if (print_info) {
        fmt::print("Read {} data points with {} features in {} using the arff parser.\n",
                   data_ptr->size(),
                   max_size - 1,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

template <typename T>
void parameter<T>::parse_model_file(const std::string &) {
    // TODO: implement
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
    return out << fmt::format(
               "kernel_type      {}\n"
               "degree           {}\n"
               "gamma            {}\n"
               "coef0            {}\n"
               "cost             {}\n"
               "epsilon          {}\n"
               "print_info       {}\n"
               "backend          {}\n"
               "target platform  {}\n"
               "input_filename   {}\n"
               "model_filename   {}\n"
               "predict_filename {}\n"
               "real_type        {}\n",
               params.kernel,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               params.epsilon,
               params.print_info,
               params.backend,
               params.target,
               params.input_filename,
               params.model_filename,
               params.predict_filename,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}
template std::ostream &operator<<(std::ostream &, const parameter<float> &);
template std::ostream &operator<<(std::ostream &, const parameter<double> &);

// explicitly instantiate template class
template class parameter<float>;
template class parameter<double>;

}  // namespace plssvm
