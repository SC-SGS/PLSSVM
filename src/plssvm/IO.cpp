#include "plssvm/CSVM.hpp"

#include "plssvm/detail/file_reader.hpp"     // plssvm::detail::file_reader
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::convert_to, plssvm::detail::starts_with, plssvm::detail::ends_with, plssvm::detail::trim_left
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/typedef.hpp"                // plssvm::real_t

#include "fmt/core.h"  // fmt::format, fmt::print, fmt::format_to

#include <algorithm>    // std::max, std::transform
#include <cctype>       // std::toupper
#include <cstddef>      // std::size_t
#include <exception>    // std::exception_ptr, std::exception, std::current_exception, std::rethrow_exception
#include <iterator>     // std::back_inserter
#include <omp.h>        // omp_get_num_threads # TODO: get rid of it?
#include <ostream>      // std::ofstream, std::ios::out, std::ios::trunc
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::move
#include <vector>       // std::vector

namespace plssvm {

// read libsvm file
void CSVM::libsvmParser(const std::string &filename) {
    detail::file_reader f{ filename, '#' };

    value.resize(f.num_lines());
    data.resize(f.num_lines());

    std::size_t max_size = 0;
    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for reduction(max:max_size)
        for (std::size_t i = 0; i < data.size(); ++i) {
            #pragma omp cancellation point for
            try {
                std::string_view line = f.line(i);

                // get class
                std::size_t pos = line.find_first_of(' ');
                value[i] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(0, pos)) > real_t{ 0.0 } ? 1 : -1;
                // value[i] = std::copysign(1.0, detail::convert_to<real_t>(line.substr(0, pos)));

                // get data
                std::vector<real_t> vline(max_size);
                while (true) {
                    std::size_t next_pos = line.find_first_of(':', pos);
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
                    vline[index] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
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
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i].resize(max_size);
    }

    // update values
    num_data_points = data.size();
    num_features = max_size;

    // no features were parsed -> invalid file
    if (num_features == 0) {
        throw invalid_file_format_exception{ fmt::format("Can't parse file '{}'!", filename) };
    }

    // update gamma
    if (gamma == 0) {
        gamma = 1. / num_features;
    }

    fmt::print("Read {} data points with {} features.\n", num_data_points, num_features);
}

// read ARFF file
void CSVM::arffParser(const std::string &filename) {
    detail::file_reader f{ filename, '%' };
    std::size_t max_size = 0;

    // parse arff header
    std::size_t header = 0;
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

    value.resize(f.num_lines() - (header + 1));
    data.resize(f.num_lines() - (header + 1));

    #pragma omp parallel for
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i].resize(max_size - 1);
    }

    std::exception_ptr parallel_exception;

    #pragma omp parallel
    {
        #pragma omp for
        for (std::size_t i = 0; i < data.size(); ++i) {
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
                    std::size_t pos = 1;
                    while (true) {
                        std::size_t next_pos = line.find_first_of(' ', pos);
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
                            value[i] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(pos)) > real_t{ 0.0 } ? 1 : -1;
                        } else {
                            data[i][index] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
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
                    std::size_t pos = 0;
                    for (std::size_t j = 0; j < max_size - 1; ++j) {
                        std::size_t next_pos = line.find_first_of(',', pos);
                        if (next_pos == std::string_view::npos) {
                            throw invalid_file_format_exception{ fmt::format("Invalid number of features! Found {} but should be {}.", j, max_size - 1) };
                        }
                        data[i][j] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                        pos = next_pos + 1;
                    }
                    value[i] = detail::convert_to<real_t, invalid_file_format_exception>(line.substr(pos)) > real_t{ 0.0 } ? 1 : -1;
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
    num_data_points = data.size();
    num_features = max_size - 1;

    // update gamma
    if (gamma == 0) {
        gamma = 1. / num_features;
    }

    fmt::print("Read {} data points with {} features.\n", num_data_points, num_features);
}

void CSVM::writeModel(const std::string &model_name) {
    // TODO: idea: save number of Datapoint in input file -> copy input file -> manipulate copy and dont rewrite whole File
    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (std::size_t i = 0; i < alpha.size(); ++i) {
        if (value[i] > 0) {
            ++count_pos;
        }
        if (value[i] < 0) {
            ++count_neg;
        }
        if (alpha[i] == cost) {
            ++nBSV;
        }
    }

    // terminal output
    if (info) {
        fmt::print(
            "Optimization finished \n"
            "nu = {}\n"
            "obj = \t, rho {}\n"
            "nSV = {}, nBSV = {}\n"
            "Total nSV = {}\n",
            cost,
            -bias,
            count_pos + count_neg - nBSV,
            nBSV,
            count_pos + count_neg);
    }

    // create model file
    std::ofstream model{ model_name.data(), std::ios::out | std::ios::trunc };
    model << fmt::format(
        "svm_type c_svc\n"
        "kernel_type {}\n"
        "nr_class 2\n"
        "total_sv {}\n"
        "rho {}\n"
        "label 1 -1\n"
        "nr_sv {} {}\n"
        "SV\n",
        static_cast<plssvm::kernel_type>(kernel),
        count_pos + count_neg,
        -bias,
        count_pos,
        count_neg);

    volatile int count = 0;
    #pragma omp parallel
    {
        std::string out_pos;
        std::string out_neg;

        // all support vectors with class 1
        #pragma omp for nowait
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            if (value[i] > 0) {
                fmt::format_to(std::back_inserter(out_pos), "{} ", alpha[i]);
                for (std::size_t j = 0; j < data[i].size(); ++j) {
                    if (data[i][j] != 0) {
                        fmt::format_to(std::back_inserter(out_pos), "{}:{:e} ", j, data[i][j]);
                    }
                }
                out_pos.push_back('\n');
            }
        }

        #pragma omp critical
        {
            model.write(out_pos.data(), static_cast<std::streamsize>(out_pos.size()));
            count++;
            #pragma omp flush(count, model)
        }

        // all support vectors with class -1
        #pragma omp for nowait
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            if (value[i] < 0) {
                fmt::format_to(std::back_inserter(out_neg), "{} ", alpha[i]);
                for (std::size_t j = 0; j < data[i].size(); ++j) {
                    if (data[i][j] != 0) {
                        fmt::format_to(std::back_inserter(out_neg), "{}:{:e} ", j, data[i][j]);
                    }
                }
                out_neg.push_back('\n');
            }
        }

        // wait for all threads to write support vectors for class 1
        while (count < omp_get_num_threads()) {
        }

        #pragma omp critical
        model.write(out_neg.data(), static_cast<std::streamsize>(out_neg.size()));
    }
    model.close();
}

}  // namespace plssvm
