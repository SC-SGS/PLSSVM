#include <plssvm/CSVM.hpp>
#include <plssvm/exceptions.hpp>
#include <plssvm/operators.hpp>
#include <plssvm/string_utility.hpp>

#include <fmt/core.h>

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
            } else if (util::starts_with(trimmed, "@RELATION")) {
                // ignore relation
                continue;
            } else if (util::starts_with(trimmed, "@ATTRIBUTE")) {
                // toupper all letters
                std::transform(line.begin(), line.end(), line.begin(), [](const char c) { return std::toupper(c); });
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
}

void CSVM::writeModel(const std::string_view model_name) { //TODO: idea: save number of Datapoint in input file ->  copy input file -> manipulate copy and dont rewrite whole File
    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (int i = 0; i < alpha.size(); ++i) {
        if (value[i] > 0)
            ++count_pos;
        if (value[i] < 0)
            ++count_neg;
        if (alpha[i] == cost)
            ++nBSV;
    }
    //Terminal output
    if (info) {
        std::cout << "Optimization finished \n";
        std::cout << "nu = " << cost << "\n";
        std::cout << "obj = "
                  << "\t"
                  << ", rho " << -bias << "\n";
        std::cout << "nSV = " << count_pos + count_neg - nBSV << ", nBSV = " << nBSV << "\n";
        std::cout << "Total nSV = " << count_pos + count_neg << std::endl;
    }

    //Model file
    std::ofstream model(model_name.data(), std::ios::out | std::ios::trunc);
    model << "svm_type c_svc\n";
    switch (kernel) {
    case 0:
        model << "kernel_type "
              << "linear"
              << "\n";
        break;
    case 1:
        model << "kernel_type "
              << "polynomial"
              << "\n";
        break;
    case 2:
        model << "kernel_type "
              << "rbf"
              << "\n";
        break;
    default:
        throw std::runtime_error("Can not decide which kernel!");
    }
    model << "nr_class 2\n";
    model << "total_sv " << count_pos + count_neg << "\n";
    model << "rho " << -bias << "\n";
    model << "label "
          << "1"
          << " "
          << "-1"
          << "\n";
    model << "nr_sv " << count_pos << " " << count_neg << "\n";
    model << "SV\n";
    model << std::scientific;

    volatile int count = 0;

    #pragma omp parallel //num_threads(1) //TODO: fix bug. SV complete missing if more then 1 Thread
    {
        std::stringstream out_pos;
        std::stringstream out_neg;

        // Alle SV Klasse 1
        #pragma omp for nowait
        for (int i = 0; i < alpha.size(); ++i) {
            if (value[i] > 0)
                out_pos << alpha[i] << " " << data[i] << "\n";
        }

        #pragma omp critical
        {
            model << out_pos.str();
            count++;
            #pragma omp flush(count, model)
        }

        // Alle SV Klasse -1
        #pragma omp for nowait
        for (int i = 0; i < alpha.size(); ++i) {
            if (value[i] < 0)
                out_neg << alpha[i] << " " << data[i] << "\n";
        }

        //Wait for all have written Class 1
        while (count < omp_get_num_threads()) {
        };

        #pragma omp critical
        model << out_neg.str();
    }
    model.close();
}

} // namespace plssvm
