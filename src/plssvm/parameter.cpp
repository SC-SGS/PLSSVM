/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "fmt/core.h"     // fmt::format, fmt::print
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <iostream>  // std::ostream

namespace plssvm {

/*
template <typename T>
void parameter<T>::parse_model_file(const std::string &filename) {
    auto start_time = std::chrono::steady_clock::now();

    // set new filenames
    if (predict_filename == predict_name_from_input() || predict_filename.empty()) {
        model_filename = filename;
        predict_filename = predict_name_from_input();
    }
    model_filename = filename;

    detail::io::file_reader f{ filename, '#' };

    // reset values pointer
    value_ptr = nullptr;

    // helper variables
    unsigned long long num_sv{ 0 };
    std::pair labels{ real_type{ 0.0 }, real_type{ 0.0 } };
    bool rho_set{ false };

    // parse libsvm model file header
    std::size_t header = 0;
    {
        for (; header < f.num_lines(); ++header) {
            std::string line{ detail::trim(f.line(header)) };
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
                const auto nr_class = detail::convert_to<unsigned int>(value);
                if (nr_class != 2) {
                    throw invalid_file_format_exception{ fmt::format("Can only use 2 classes, but {} were given!", nr_class) };
                }
            } else if (detail::starts_with(line, "total_sv")) {
                // the total number of support vectors must be greater than 0
                num_sv = detail::convert_to<decltype(num_sv)>(value);
                if (num_sv == 0) {
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
                labels.second = detail::convert_to<real_type, invalid_file_format_exception>(second_label);
                value.remove_prefix(std::min(second_label.size() + 1, value.size()));

                value = detail::trim_left(value);

                if (!value.empty() || (labels.first != 1 && labels.first != -1) || (labels.second != 1 && labels.second != -1)) {
                    throw invalid_file_format_exception{ fmt::format("Only the labels 1 and -1 are allowed, but '{}' were given!", line) };
                }
            } else if (detail::starts_with(line, "nr_sv")) {
                // parse first number
                const std::string_view first_num = value.substr(0, value.find_first_of(' '));
                const auto num_first = detail::convert_to<unsigned long long>(first_num);
                value.remove_prefix(std::min(first_num.size() + 1, value.size()));
                // parse second number
                const std::string_view second_num = value.substr(0, value.find_first_of(" \n"));
                const auto num_second = detail::convert_to<unsigned long long>(second_num);
                value.remove_prefix(std::min(second_num.size() + 1, value.size()));

                value = detail::trim_left(value);

                if (!value.empty()) {
                    // error if more than two numbers were given
                    throw invalid_file_format_exception{ fmt::format("Only two numbers are allowed, but more were given '{}'!", line) };
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
                break;
            } else {
                throw invalid_file_format_exception{ fmt::format("Unrecognized header entry '{}'! Maybe SV is missing?", f.line(header)) };
            }
        }
    }

    // additional sanity checks
    if (num_sv == 0) {
        // no total number of support vectors given
        throw invalid_file_format_exception{ "Missing total number of support vectors!" };
    } else if (labels.first == 0 || labels.second == 0) {
        // no labels given
        throw invalid_file_format_exception{ "Missing labels!" };
    } else if (value_ptr == nullptr) {
        // no count for positive and negative support vectors given
        throw invalid_file_format_exception{ "Missing number of support vectors per class!" };
    } else if (!rho_set) {
        // no rho set
        throw invalid_file_format_exception{ "Missing rho value!" };
    } else if (header + 1 >= f.num_lines()) {
        // no support vectors given
        throw invalid_file_format_exception{ "Can't parse file: no support vectors are given or SV is missing!" };
    }

    // parse support vectors
    std::vector<std::vector<real_type>> data(num_sv);
    std::vector<real_type> alphas(num_sv);

    // parse support vectors
    detail::parse_libsvm_content(f, header + 1, data, alphas);

    // update shared pointer
    data_ptr = std::make_shared<const std::vector<std::vector<real_type>>>(std::move(data));
    alpha_ptr = std::make_shared<const std::vector<real_type>>(std::move(alphas));

    auto end_time = std::chrono::steady_clock::now();
    if (print_info) {
        fmt::print("Read {} support vectors with {} features in {} using the libsvm model parser from file '{}'.\n",
                   data_ptr->size(),
                   (*data_ptr)[0].size(),
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time),
                   filename);
    }
}
*/


// explicitly instantiate template class
//template class parameter<float>;
//template class parameter<double>;
//
//
//template <typename T>
//std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
//    return out << fmt::format(
//               "kernel_type                 {}\n"
//               "degree                      {}\n"
//               "gamma                       {}\n"
//               "coef0                       {}\n"
//               "cost                        {}\n"
//               "epsilon                     {}\n"
//               "backend                     {}\n"
//               "target platform             {}\n"
//               "SYCL kernel invocation type {}\n"
//               "SYCL implementation type    {}\n"
//               "real_type                   {}\n",
//               params.kernel,
//               params.degree,
//               params.gamma,
//               params.coef0,
//               params.cost,
//               params.epsilon,
//               params.backend,
//               params.target,
//               params.sycl_kernel_invocation_type,
//               params.sycl_implementation_type,
//               detail::arithmetic_type_name<typename parameter<T>::real_type>());
//}
//template std::ostream &operator<<(std::ostream &, const parameter<float> &);
//template std::ostream &operator<<(std::ostream &, const parameter<double> &);

}  // namespace plssvm
