#include <plssvm/CSVM.hpp>
#include <plssvm/exceptions.hpp>
#include <plssvm/operators.hpp>
#include <plssvm/string_utility.hpp>

#include <fmt/format.h>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace plssvm {

// read libsvm file
void CSVM::libsvmParser(const std::string_view filename) {
    std::vector<std::string> data_lines;

    {
        std::ifstream file{filename.data()};
        if (file.fail()) {
            throw file_not_found_exception{fmt::format("Couldn't find file: '{}'!", filename)};
        }
        std::string line;
        while (std::getline(file, line)) {
            std::string_view trimmed = util::trim_left(line);
            if (!trimmed.empty() && !util::starts_with(trimmed, '#')) {
                data_lines.push_back(std::move(line));
            }
        }
    }

    value.resize(data_lines.size());
    data.resize(data_lines.size());

    std::size_t max_size = 0;

    #pragma omp parallel for reduction(max:max_size)
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::string_view line = data_lines[i];

        // get class
        std::size_t pos = line.find_first_of(' ');
        value[i] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(0, pos)) > real_t{0.0} ? 1 : -1;
        // value[i] = std::copysign(1.0, util::convert_to<real_t>(line.substr(0, pos)));

        // get data
        std::vector<real_t> vline(max_size);
        std::size_t next_pos = line.find_first_of(':', pos);
        while (next_pos != std::string_view::npos) {
            // get index
            const auto index = util::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
            if (index >= vline.size()) {
                vline.resize(index + 1);
            }
            pos = next_pos + 1;

            // get value
            next_pos = line.find_first_of(',', pos);
            vline[index] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));

            if (next_pos == std::string_view::npos) {
                break;
            }
            pos = next_pos + 1;
            next_pos = line.find_first_of(':', pos);
        }
        max_size = std::max(max_size, vline.size());
        data[i] = std::move(vline);
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
        throw invalid_file_format_exception{fmt::format("Can't parse file '{}'!", filename)};
    }

    // update gamma
    if (gamma == 0) {
        gamma = 1. / num_features;
    }

    fmt::print("Read {} data points with {} features.\n", num_data_points, num_features);
}

// read ARF file
void CSVM::arffParser(const std::string_view filename) {
    std::vector<std::string> data_lines;
    std::size_t max_size = 0;
    {
        std::ifstream file{filename.data()};
        if (file.fail()) {
            throw file_not_found_exception{fmt::format("Couldn't find file: '{}'!", filename)};
        }
        std::string line;

        // parse header information
        while (std::getline(file, line)) {
            std::string_view trimmed = util::trim_left(line);
            if (trimmed.empty() || util::starts_with(trimmed, '%')) {
                // ignore empty lines or comments
                continue;
            } else {
                // match arff attribute case insensitive
                std::transform(line.begin(), line.end(), line.begin(), [](const char c) { return std::toupper(c); });
                trimmed = util::trim_left(line);
                if (util::starts_with(trimmed, "@RELATION")) {
                    // ignore relation
                    continue;
                } else if (util::starts_with(trimmed, "@ATTRIBUTE")) {
                    if (line.find("NUMERIC") == std::string::npos) {
                        throw invalid_file_format_exception{fmt::format("Can only use NUMERIC features, but '{}' was given!", line)};
                    }
                    // add a feature
                    ++max_size;
                } else if (util::starts_with(trimmed, "@DATA")) {
                    // finished reading header -> start parsing data
                    break;
                }
            }
        }

        // something went wrong, e.g., no @ATTRIBUTE fields
        if (max_size == 0) {
            throw invalid_file_format_exception{"Invalid file format!"};
        }

        // parse data
        while (std::getline(file, line)) {
            std::string_view trimmed = util::trim_left(line);
            if (trimmed.empty() || util::starts_with(trimmed, '%')) {
                // ignore empty lines or comments
                continue;
            } else if (util::starts_with(trimmed, '@')) {
                // read @ inside data section
                throw invalid_file_format_exception{fmt::format("Read @ inside data section!: {}", line)};
            } else {
                data_lines.push_back(std::move(line));
            }
        }
    }

    value.resize(data_lines.size());
    data.resize(data_lines.size());

    #pragma omp parallel for
    for (std::size_t i = 0; i < data_lines.size(); ++i) {
        data[i].resize(max_size - 1);
    }

    #pragma omp parallel for
    for (std::size_t i = 0; i < data.size(); ++i) { // TODO: exceptions
        std::string_view line{util::trim_left(data_lines[i])};

        if (util::starts_with(line, '{')) {
            // missing closing }
            if (!util::ends_with(line, '}')) {
                throw invalid_file_format_exception{"Missing closing '}' for sparse data point description!"};
            }
            // sparse line
            bool is_class_set = false;
            std::size_t pos = 1;
            std::size_t next_pos = line.find_first_of(' ', pos);
            while (next_pos != std::string_view::npos) {
                // get index
                const auto index = util::convert_to<unsigned long, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                pos = next_pos + 1;

                // get value
                next_pos = line.find_first_of(",}", pos);

                // write parsed value depending on the index
                if (index == max_size - 1) {
                    is_class_set = true;
                    value[i] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos)) > real_t{0.0} ? 1 : -1;
                } else {
                    data[i][index] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                }

                if (next_pos == std::string_view::npos) {
                    break;
                }
                // remove already processes part of the line
                line.remove_prefix(next_pos + 1);
                line = util::trim_left(line);
                pos = 0;
                next_pos = line.find_first_of(' ');
            }
            // no class label found
            if (!is_class_set) {
                throw invalid_file_format_exception{"Missing class for data point!"};
            }
        } else {
            // dense line
            std::size_t pos = 0;
            for (std::size_t j = 0; j < max_size - 1; ++j) {
                std::size_t next_pos = line.find_first_of(',', pos);
                if (next_pos == std::string_view::npos) {
                    throw invalid_file_format_exception{fmt::format("Invalid number of features! Found {} but should be {}.", j, max_size - 1)};
                }
                data[i][j] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos, next_pos - pos));
                pos = next_pos + 1;
            }
            value[i] = util::convert_to<real_t, invalid_file_format_exception>(line.substr(pos)) > real_t{0.0} ? 1 : -1;
        }
    }

    num_data_points = data.size();
    num_features = max_size - 1;

    if (gamma == 0) {
        gamma = 1. / num_features;
    }

    fmt::print("Read {} data points with {} features.\n", num_data_points, num_features);
}

void CSVM::writeModel(const std::string_view model_name) { //TODO: idea: save number of Datapoint in input file ->  copy input file -> manipulate copy and dont rewrite whole File
    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (int i = 0; i < alpha.size(); ++i) {
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
            cost, -bias, count_pos + count_neg - nBSV, nBSV, count_pos + count_neg);
    }

    // function to get the kernel name from its ID
    auto kernel_to_string = [](const auto k) -> std::string_view {
        switch (k) {
        case 0:
            return "linear";
        case 1:
            return "polynomial";
        case 2:
            return "rbf";
        default:
            throw std::runtime_error{fmt::format("Invalid kernel value {}!", k)};
        }
    };

    // create model file
    std::ofstream model{model_name.data(), std::ios::out | std::ios::trunc};
    model << fmt::format(
        "svm_type c_svc\n"
        "kernel_type {}\n"
        "nr_class 2\n"
        "total_sv {}\n"
        "rho {}\n"
        "label 1 -1\n"
        "nr_sv {} {}\n"
        "SV\n",
        kernel_to_string(kernel), count_pos + count_neg, -bias, count_pos, count_neg);

    volatile int count = 0;
    #pragma omp parallel
    {
        fmt::memory_buffer out_pos;
        fmt::memory_buffer out_neg;

        // all support vectors with class 1
        #pragma omp for nowait
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            if (value[i] > 0) {
                fmt::format_to(out_pos.begin(), "{}", alpha[i]);
                for (std::size_t j = 0; j < data[i].size(); ++j) {
                    if (data[i][j] != 0) {
                        fmt::format_to(out_pos.begin(), "{}:{:e} ", j, data[i][j]);
                    }
                }
                fmt::format_to(out_pos.begin(), "\n");
            }
        }

        #pragma omp critical
        {
            model << fmt::to_string(out_pos);
            count++;
            #pragma omp flush(count, model)
        }

        // all support vectors with class -1
        #pragma omp for nowait
        for (std::size_t i = 0; i < alpha.size(); ++i) {
            if (value[i] < 0) {
                fmt::format_to(out_neg.begin(), "{}", alpha[i]);
                for (std::size_t j = 0; j < data[i].size(); ++j) {
                    if (data[i][j] != 0) {
                        fmt::format_to(out_neg.begin(), "{}:{:e} ", j, data[i][j]);
                    }
                }
                fmt::format_to(out_neg.begin(), "\n");
            }
        }

        // wait for all threads to write support vectors for class 1
        while (count < omp_get_num_threads()) {
        }

        #pragma omp critical
        model << fmt::to_string(out_neg);
    }
    model.close();
}

} // namespace plssvm
