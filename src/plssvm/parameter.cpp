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
#include <sstream>      // std::istringstream
#include <string>       // std::string, std::getline
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
    if (model_filename == model_name_from_input() || model_filename.empty()) {
        input_filename = filename;
        model_filename = model_name_from_input();
    }
    input_filename = filename;
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
    if (model_filename == model_name_from_input() || model_filename == "") {
        input_filename = filename;
        model_filename = model_name_from_input();
    }
    input_filename = filename;
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
void parameter<T>::parse_model_file(const std::string &filename) {
    auto start_time = std::chrono::steady_clock::now();

    detail::file_reader f{ filename, '#' };

    // reset data
    data_ptr = nullptr;
    value_ptr = nullptr;
    alphas_ptr = nullptr;

    // helper variables
    size_type num_sv{ 0 };
    std::pair<int, int> labels{ 0, 0 };
    bool rho_set{ false };
    bool sv_set{ false };

    // parse libsvm model file header
    size_type header = 0;
    {
        for (; header < f.num_lines(); ++header) {
            std::string line{ f.line(header) };
            std::transform(line.begin(), line.end(), line.begin(), [](const char c) { return std::tolower(c); });

            // separate value from model header entry
            std::string_view value{ line };
            value.remove_prefix(std::min(value.find_first_of(' ') + 1, value.size()));
            value = detail::trim_left(value);

            if (detail::starts_with(line, "svm_type")) {
                // svm_type must be c_svc
                if (value != "c_svc") {
                    throw invalid_file_format_exception{ fmt::format("Can only use c_svc as svm_type, but '{}' was given!", line) };
                }
            } else if (detail::starts_with(line, "kernel_type")) {
                // parse kernel_type, must be linear, polynomial or rbf
                std::istringstream iss{ std::string{ value } };
                iss >> kernel;
                if (iss.fail()) {
                    throw invalid_file_format_exception{ fmt::format("Unrecognized kernel type '{}'!", value) };
                }
            } else if (detail::starts_with(line, "gamma")) {
                // parse gamma
                gamma = detail::convert_to<decltype(gamma)>(value);
            } else if (detail::starts_with(line, "degree")) {
                // parse degree
                degree = detail::convert_to<decltype(degree)>(value);
            } else if (detail::starts_with(line, "coef0")) {
                // parse coef0
                coef0 = detail::convert_to<decltype(coef0)>(value);
            } else if (detail::starts_with(line, "nr_class")) {
                // number of classes must be 2
                const auto nr_class = detail::convert_to<size_type>(value);
                if (nr_class != 2) {
                    throw invalid_file_format_exception{ fmt::format("Can only use 2 classes, but {} were given!", nr_class) };
                }
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_sv = detail::convert_to<size_type>(value);
                if (num_sv <= 0) {
                    throw invalid_file_format_exception{ fmt::format("The number of support vectors must be greater than 0, but is {}!", num_sv) };
                }
            } else if (detail::starts_with(line, "rho")) {
                // parse rho, required
                rho = detail::convert_to<real_type>(value);
                rho_set = true;
            } else if (detail::starts_with(line, "label")) {
                // parse first label
                const std::string_view first_label = value.substr(0, value.find_first_of(' '));
                labels.first = detail::convert_to<real_type>(first_label);
                value.remove_prefix(std::min(first_label.size() + 1, value.size()));
                // parse second label
                const std::string_view second_label = value.substr(0, value.find_first_of(" \n"));
                labels.second = detail::convert_to<real_type>(second_label);
                value.remove_prefix(std::min(second_label.size() + 1, value.size()));

                value = detail::trim_left(value);

                if (!value.empty() || (labels.first != 1 && labels.first != -1) || (labels.second != 1 && labels.second != -1)) {
                    throw invalid_file_format_exception{ fmt::format("Only the labels 1 and -1 are allowed, but '{}' were given!", line) };
                }
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse first number
                const std::string_view first_num = value.substr(0, value.find_first_of(' '));
                const auto num_first = detail::convert_to<size_type>(first_num);
                value.remove_prefix(std::min(first_num.size() + 1, value.size()));
                // parse second number
                const std::string_view second_num = value.substr(0, value.find_first_of(" \n"));
                const auto num_second = detail::convert_to<real_type>(second_num);
                value.remove_prefix(std::min(second_num.size() + 1, value.size()));

                value = detail::trim_left(value);

                if (!value.empty()) {
                    // error if more than two numbers were given
                    throw invalid_file_format_exception{ fmt::format("Only two numbers are allowed, but more were given {}!", line) };
                } else if (num_first + num_second != num_sv) {
                    throw invalid_file_format_exception{ fmt::format("The number of positive and negative support vectors doesn't add up to the total number: {} + {} != {}!", num_first, num_second, num_sv) };
                }

                // update values
                std::vector<real_type> values(num_sv);
                std::fill(values.begin(), values.begin() + num_first, labels.first);
                std::fill(values.begin() + num_first, values.end(), labels.second);
                value_ptr = std::make_shared<const std::vector<real_type>>(std::move(values));
            } else if (line == "sv") {
                // start parsing support vectors, required
                sv_set = true;
                break;
            } else {
                throw invalid_file_format_exception{ fmt::format("Unrecognized header entry '{}'!", line) };
            }
        }
    }

    // additional sanity checks
    if (num_sv == 0) {
        // no total number of support vectors given
        throw invalid_file_format_exception{ "Missing number of support vectors!" };
    } else if (labels.first == 0 || labels.second == 0) {
        // no labels given
        throw invalid_file_format_exception{ "Missing labels!" };
    } else if (value_ptr == nullptr) {
        // no count for positive and negative support vectors given
        throw invalid_file_format_exception{ "Missing number of support vectors per class!" };
    } else if (!rho_set) {
        // no rho set
        throw invalid_file_format_exception{ "Missing rho value!" };
    } else if (!sv_set) {
        // no support vectors given
        throw invalid_file_format_exception{ "Missing support vectors!" };
    }

    // parse support vectors
    std::vector<std::vector<real_type>> data(num_sv);
    std::vector<real_type> alphas(num_sv);

    size_type max_size = 0;
    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for reduction(max : max_size)
        for (size_type i = 0; i < data.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = f.line(i + header + 1);

                // get alpha
                size_type pos = line.find_first_of(' ');
                alphas[i] = detail::convert_to<real_type, invalid_file_format_exception>(line.substr(0, pos));

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

    // update shared pointer
    data_ptr = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data));
    alphas_ptr = std::make_shared<const std::vector<real_type>>(std::move(alphas));

    auto end_time = std::chrono::steady_clock::now();
    if (print_info) {
        fmt::print("Read {} data points with {} features in {} using the arff parser.\n",
                   data_ptr->size(),
                   max_size,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
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
               "rho              {}\n"
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
               params.rho,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}
template std::ostream &operator<<(std::ostream &, const parameter<float> &);
template std::ostream &operator<<(std::ostream &, const parameter<double> &);

template <typename T>
std::string parameter<T>::model_name_from_input() {
    std::size_t pos = input_filename.find_last_of("/\\");
    return input_filename.substr(pos + 1) + ".model";
}

// explicitly instantiate template class
template class parameter<float>;
template class parameter<double>;

}  // namespace plssvm
