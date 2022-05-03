/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/csvm.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"       // dot product, plssvm::operators::sum, plssvm::operators::sign
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception, plssvm::exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type, plssvm::kernel_function
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "plssvm/constants.hpp"

#include "fmt/chrono.h"   // format std::chrono
#include "fmt/compile.h"  // FMT_COMPILE
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/os.h"       // fmt::output_file
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#ifdef _OPENMP
    #include <omp.h>  // omp_get_num_threads
#endif

#include <algorithm>  // std::all_of
#include <chrono>     // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <cstddef>    // std::size_t
#include <fstream>    // std::ofstream
#include <ios>        // std:streamsize, std::ios
#include <memory>     // std::make_shared
#include <string>     // std::string
#include <utility>    // std::move
#include <vector>     // std::vector

namespace plssvm {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    target_{ params.target }, kernel_{ params.kernel }, degree_{ params.degree }, gamma_{ params.gamma }, coef0_{ params.coef0 }, cost_{ params.cost }, epsilon_{ params.epsilon }, print_info_{ plssvm::verbose }, data_ptr_{ nullptr }, value_ptr_{ nullptr }, alpha_ptr_{ nullptr }, bias_{ -0.0 } {
    if (data_ptr_ == nullptr) {
        throw exception{ "No data points provided!" };
    } else if (data_ptr_->empty()) {
        throw exception{ "Data set is empty!" };
    } else if (!std::all_of(data_ptr_->begin(), data_ptr_->end(), [&](const std::vector<real_type> &point) { return point.size() == data_ptr_->front().size(); })) {
        throw exception{ "All points in the data vector must have the same number of features!" };
    } else if (data_ptr_->front().empty()) {
        throw exception{ "No features provided for the data points!" };
    } else if (alpha_ptr_ != nullptr && alpha_ptr_->size() != data_ptr_->size()) {
        throw exception{ fmt::format("Number of weights ({}) must match the number of data points ({})!", alpha_ptr_->size(), data_ptr_->size()) };
    }

    num_data_points_ = data_ptr_->size();
    num_features_ = (*data_ptr_)[0].size();
}

template <typename T>
void csvm<T>::write_model(const std::string &model_name) {
    auto start_time = std::chrono::steady_clock::now();

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor

    if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas given! Maybe a call to 'learn()' is missing?" };
    } else if (value_ptr_ == nullptr) {
        throw exception{ "No labels given! Maybe the data is only usable for prediction?" };
    } else if (data_ptr_->size() != value_ptr_->size()) {
        throw exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", value_ptr_->size(), data_ptr_->size()) };
    }
    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");                                                                    // exception in constructor
    PLSSVM_ASSERT(std::all_of(data_ptr_->begin(), data_ptr_->end(), [&](const std::vector<real_type> &point) { return point.size() == data_ptr_->front().size(); }),
                  "All points in the data vector must have the same number of features!");    // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->front().empty(), "No features provided for the data points!");  // exception in constructor

    unsigned long long nBSV{ 0 };
    unsigned long long count_pos{ 0 };
    unsigned long long count_neg{ 0 };
    for (typename std::vector<real_type>::size_type i = 0; i < alpha_ptr_->size(); ++i) {
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
    auto model = fmt::output_file(model_name);
    model.print("{}", libsvm_model_header);

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

    volatile int count = 0;
    #pragma omp parallel
    {
        // all support vectors with class 1
        std::string out_pos;
        #pragma omp for nowait
        for (typename std::vector<real_type>::size_type i = 0; i < alpha_ptr_->size(); ++i) {
            if ((*value_ptr_)[i] > 0) {
                format_libsvm_line(out_pos, (*alpha_ptr_)[i], (*data_ptr_)[i]);
            }
        }

        #pragma omp critical
        {
            model.print("{}", out_pos);
            count++;
            #pragma omp flush(count, model)
        }

        // all support vectors with class -1
        std::string out_neg;
        #pragma omp for nowait
        for (typename std::vector<real_type>::size_type i = 0; i < alpha_ptr_->size(); ++i) {
            if ((*value_ptr_)[i] < 0) {
                format_libsvm_line(out_neg, (*alpha_ptr_)[i], (*data_ptr_)[i]);
            }
        }

        // wait for all threads to write support vectors for class 1
#ifdef _OPENMP
        while (count < omp_get_num_threads()) {
        }
#else
        #pragma omp barrier
#endif

        #pragma omp critical
        model.print("{}", out_neg);
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Wrote model file ('{}') with {} support vectors in {}.\n",
                   model_name,
                   count_pos + count_neg,
                   std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

template <typename T>
void csvm<T>::learn() {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor

    if (value_ptr_ == nullptr) {
        throw exception{ "No labels given for training! Maybe the data is only usable for prediction?" };
    } else if (data_ptr_->size() != value_ptr_->size()) {
        throw exception{ fmt::format("Number of labels ({}) must match the number of data points ({})!", value_ptr_->size(), data_ptr_->size()) };
    }

    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");  // exception in constructor
    PLSSVM_ASSERT(std::all_of(data_ptr_->begin(), data_ptr_->end(), [&](const std::vector<real_type> &point) { return point.size() == data_ptr_->front().size(); }),
                  "All points in the data vector must have the same number of features!");    // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->front().empty(), "No features provided for the data points!");  // exception in constructor

    // setup the data on the device
    setup_data_on_device();

    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> q;
    std::vector<real_type> b = *value_ptr_;
    #pragma omp parallel sections
    {
        #pragma omp section  // generate q
        {
            q = generate_q();
        }
        #pragma omp section  // generate right-hand side from equation
        {
            b.pop_back();
            b -= value_ptr_->back();
        }
        #pragma omp section  // generate bottom right from A
        {
            QA_cost_ = kernel_function(data_ptr_->back(), data_ptr_->back()) + 1 / cost_;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Setup for solving the optimization problem done in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    start_time = std::chrono::steady_clock::now();

    // solve minimization
    std::vector<real_type> alpha;
    alpha = solver_CG(b, num_features_, epsilon_, q);
    bias_ = value_ptr_->back() + QA_cost_ * sum(alpha) - (transposed{ q } * alpha);
    alpha.emplace_back(-sum(alpha));

    alpha_ptr_ = std::make_shared<const std::vector<real_type>>(std::move(alpha));
    w_.clear();

    end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Solved minimization problem (r = b - Ax) using CG in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

template <typename T>
auto csvm<T>::accuracy() -> real_type {
    if (value_ptr_ == nullptr) {
        throw exception{ "No labels given! Maybe the data is only usable for prediction?" };
    }
    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor

    return accuracy(*data_ptr_, *value_ptr_);
}

template <typename T>
auto csvm<T>::accuracy(const std::vector<real_type> &point, const real_type correct_label) -> real_type {
    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor
    if (point.size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features of the predict point ({})!", data_ptr_->front().size(), point.size()) };
    }

    return accuracy(std::vector<std::vector<real_type>>(1, point), std::vector<real_type>(1, correct_label));
}

template <typename T>
auto csvm<T>::accuracy(const std::vector<std::vector<real_type>> &points, const std::vector<real_type> &correct_labels) -> real_type {
    if (points.size() != correct_labels.size()) {
        throw exception{ fmt::format("Number of data points ({}) must match number of correct labels ({})!", points.size(), correct_labels.size()) };
    }

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor

    // return zero as accuracy for predicting no points
    if (points.empty()) {
        return real_type{ 0.0 };
    }

    if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
        throw exception{ "All points in the prediction point vector must have the same number of features!" };
    } else if (points.front().size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
    }

    unsigned long long correct{ 0 };
    const std::vector<real_type> predictions = predict(points);
    for (typename std::vector<real_type>::size_type index = 0; index < predictions.size(); ++index) {
        if (predictions[index] * correct_labels[index] > real_type{ 0.0 }) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(points.size());
}

template <typename T>
auto csvm<T>::predict(const std::vector<real_type> &point) -> real_type {
    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor
    if (point.size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features of the predict point ({})!", data_ptr_->front().size(), point.size()) };
    }

    return predict(std::vector<std::vector<real_type>>(1, point))[0];
}

template <typename T>
auto csvm<T>::predict_label(const std::vector<real_type> &point) -> real_type {
    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor
    if (point.size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features of the predict point ({})!", data_ptr_->front().size(), point.size()) };
    }

    return operators::sign(predict(point));
}

template <typename T>
auto csvm<T>::predict_label(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor

    // return empty vector if there are no points to predict
    if (points.empty()) {
        return std::vector<real_type>{};
    }

    if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
        throw exception{ "All points in the prediction point vector must have the same number of features!" };
    } else if (points.front().size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
    }

    std::vector<real_type> classes(predict(points));

    // map prediction values to labels
    #pragma omp parallel for
    for (typename std::vector<real_type>::size_type i = 0; i < classes.size(); ++i) {
        classes[i] = operators::sign(classes[i]);
    }
    return classes;
}

template <typename T>
auto csvm<T>::kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj) -> real_type {
    PLSSVM_ASSERT(xi.size() == xj.size(), "Sizes mismatch!: {} != {}", xi.size(), xj.size());

    switch (kernel_) {
        case kernel_type::linear:
            return plssvm::kernel_function<kernel_type::linear>(xi, xj);
        case kernel_type::polynomial:
            return plssvm::kernel_function<kernel_type::polynomial>(xi, xj, degree_, gamma_, coef0_);
        case kernel_type::rbf:
            return plssvm::kernel_function<kernel_type::rbf>(xi, xj, gamma_);
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(kernel_)) };
}

template <typename T>
auto csvm<T>::transform_data(const std::vector<std::vector<real_type>> &matrix, const std::size_t boundary, const std::size_t num_points) -> std::vector<real_type> {
    PLSSVM_ASSERT(!matrix.empty(), "Matrix is empty!");
    PLSSVM_ASSERT(num_points <= matrix.size(), "Num points to transform can not exceed matrix size!");

    const typename std::vector<real_type>::size_type num_features = matrix[0].size();

    PLSSVM_ASSERT(std::all_of(matrix.begin(), matrix.end(), [=](const std::vector<real_type> &point) { return point.size() == num_features_; }), "Feature sizes mismatch! All features should have size {}!", num_features_);

    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> vec(num_features * (num_points + boundary));
    #pragma omp parallel for collapse(2)
    for (typename std::vector<real_type>::size_type col = 0; col < num_features; ++col) {
        for (std::size_t row = 0; row < num_points; ++row) {
            vec[col * (num_points + boundary) + row] = matrix[row][col];
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Transformed dataset from 2D AoS to 1D SoA in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
    return vec;
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm
