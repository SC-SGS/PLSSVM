/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/csvm.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::format, fmt::print

#if __has_include(<omp.h>)
    #include <omp.h>  // omp_get_num_threads
#endif

#include <chrono>   // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <fstream>  // std::ofstream, std::ios::out, std::ios::trunc
#include <ios>      // std:streamsize
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm {

template <typename T>
void csvm<T>::write_model(const std::string &model_name) {
    auto start_time = std::chrono::steady_clock::now();

    if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas given! Maybe a call to 'learn()' is missing?" };
    } else if (value_ptr_ == nullptr) {
        throw exception{ "No labels given! Maybe the data is only usable for prediction?" };
    }

    size_type nBSV = 0;
    size_type count_pos = 0;
    size_type count_neg = 0;
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
    std::string libsvm_model_header = fmt::format("svm_type c_svc\n"
                                                  "kernel_type {}\n",
                                                  kernel_);
    switch (kernel_) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            libsvm_model_header += fmt::format(
                "degree {}\n"
                "gamma {}\n"
                "coef0 {}\n",
                degree_,
                gamma_,
                coef0_);
            break;
        case kernel_type::rbf:
            libsvm_model_header += fmt::format(
                "gamma {}\n",
                gamma_);
            break;
    }
    libsvm_model_header += fmt::format(
        "nr_class 2\n"
        "total_sv {}\n"
        "rho {}\n"
        "label 1 -1\n"
        "nr_sv {} {}\n"
        "SV\n",
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
