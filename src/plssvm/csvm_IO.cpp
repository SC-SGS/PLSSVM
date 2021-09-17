/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/csvm.hpp"

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

template <typename T>
void csvm<T>::write_model(const std::string &model_name) {
    auto start_time = std::chrono::steady_clock::now();

    // TODO: asserts

    int nBSV = 0;
    int count_pos = 0;
    int count_neg = 0;
    for (size_type i = 0; i < alpha_ptr_->size(); ++i) {
        if ((*value_ptr_)[i] > 0) {
            ++count_pos;
        }
        if ((*value_ptr_)[i] < 0) {
            ++count_neg;
        }
        if ((*alpha_ptr_)[i] == cost_) {
            ++nBSV;
        }
    }

    // create libsvm model header
    std::string libsvm_model_header;
    switch (kernel_) {
        case kernel_type::linear:
            libsvm_model_header = fmt::format(
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
            break;
        case kernel_type::polynomial:
            libsvm_model_header = fmt::format(
                "svm_type c_svc\n"
                "kernel_type {}\n"
                "gamma {}\n"
                "nr_class 2\n"
                "total_sv {}\n"
                "rho {}\n"
                "label 1 -1\n"
                "nr_sv {} {}\n"
                "SV\n",
                kernel_,
                gamma_,
                count_pos + count_neg,
                -bias_,
                count_pos,
                count_neg);
            break;
        case kernel_type::rbf:
            libsvm_model_header = fmt::format(
                "svm_type c_svc\n"
                "kernel_type {}\n"
                "degree {}\n"
                "gamma {}\n"
                "coef0 {}\n"
                "nr_class 2\n"
                "total_sv {}\n"
                "rho {}\n"
                "label 1 -1\n"
                "nr_sv {} {}\n"
                "SV\n",
                kernel_,
                degree_,
                gamma_,
                coef0_,
                count_pos + count_neg,
                -bias_,
                count_pos,
                count_neg);
            break;
    }

    // terminal output
    if (print_info_) {
        fmt::print("\nOptimization finished\n{}\n", libsvm_model_header);
    }

    // create model file
    std::ofstream model{ model_name.data(), std::ios::out | std::ios::trunc };
    model << libsvm_model_header;

    // format one output-line
    auto format_libsvm_line = [](real_type a, const std::vector<real_type> &d) -> std::string {
        std::string line;
        line += fmt::format("{} ", a);
        for (size_type j = 0; j < d.size(); ++j) {
            if (d[j] != real_type{ 0.0 }) {
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
        for (size_type i = 0; i < alpha_ptr_->size(); ++i) {
            if ((*value_ptr_)[i] > 0) {
                out_pos += format_libsvm_line((*alpha_ptr_)[i], (*data_ptr_)[i]);
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
        for (size_type i = 0; i < alpha_ptr_->size(); ++i) {
            if ((*value_ptr_)[i] < 0) {
                out_neg += format_libsvm_line((*alpha_ptr_)[i], (*data_ptr_)[i]);
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
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm
