/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/csvm.hpp"

#include "plssvm/detail/operators.hpp"
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/target_platform.hpp"        // plssvm::target_platform

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::print

#include <chrono>  // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <string>  // std::string
#include <vector>  // std::vector

#include <iostream>
namespace plssvm {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    csvm{ params.target, params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
csvm<T>::csvm(const target_platform target, const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    target_{ target }, kernel_{ kernel }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 }, cost_{ cost }, epsilon_{ epsilon }, print_info_{ print_info } {}

template <typename T>
void csvm<T>::learn() {
    using namespace plssvm::operators;

    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> q;
    std::vector<real_type> b = value_;
    #pragma omp parallel sections
    {
        #pragma omp section  // generate q
        {
            q = generate_q();
        }
        #pragma omp section  // generate right-hand side from equation
        {
            b.pop_back();
            b -= value_.back();
        }
        #pragma omp section  // generate bottom right from A
        {
            QA_cost_ = kernel_function(data_.back(), data_.back()) + 1 / cost_;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Setup for solving the optimization problem done in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }

    start_time = std::chrono::steady_clock::now();

    // solve minimization
    alpha_ = solver_CG(b, num_features_, epsilon_, q);
    // old TODO: which one is correct? -> q.size() != alpha_.size() !!! -> does it have any implications on write_model?
    //    bias_ = value_.back() - QA_cost_ * alpha_.back() - (transposed{ q } * alpha_);
    // new
    bias_ = value_.back() + QA_cost_ * sum(alpha_) - (transposed{ q } * alpha_);
    alpha_.emplace_back(-sum(alpha_));

    end_time = std::chrono::steady_clock::now();
    if (print_info_) {
        fmt::print("Solved minimization problem (r = b - Ax) using CG in {}.\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
    }
}

template <typename T>
auto csvm<T>::predict(std::vector<real_type>& point) -> real_type{
    using namespace plssvm::operators;
    PLSSVM_ASSERT(data_.size() > 0, "No model or trainingsdata read");
    PLSSVM_ASSERT(data_[0].size() ==  point.size(), "Prediction point has different amount of features than training data");
    PLSSVM_ASSERT(alpha_.size() == data_.size(), "Model does not fit the training data");

    real_type temp = bias_;
    for (size_type data_index = 0; data_index < data_.size(); ++data_index){
        temp += alpha_[data_index] * kernel_function(data_[data_index], point);
    }
    // return sign(temp); // If predict should return +- 1
    return temp;
}

template <typename T>
auto csvm<T>::accuracy() -> real_type  {
    using namespace plssvm::operators;

    int correct = 0;
    for (size_type dat = 0; dat < data_.size(); ++dat){
        if ( predict(data_[dat]) * value_[dat] > 0.0){
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(data_.size());
}

template <typename T>
void csvm<T>::learn(const std::string &input_filename, const std::string &model_filename) {
    // parse data file
    parse_file(input_filename);

    // setup the data on the device
    setup_data_on_device();

    // learn model
    learn();

    // write results to model file
    write_model(model_filename);
}

template <typename T>
auto csvm<T>::kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj) -> real_type {
    switch (kernel_) {
        case kernel_type::linear:
            return plssvm::kernel_function<kernel_type::linear>(xi, xj);
        case kernel_type::polynomial:
            return plssvm::kernel_function<kernel_type::polynomial>(xi, xj, degree_, gamma_, coef0_);
        case kernel_type::rbf:
            return plssvm::kernel_function<kernel_type::rbf>(xi, xj, gamma_);
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(kernel_)) };
    }
}

template <typename T>
auto csvm<T>::transform_data(const size_type boundary) -> std::vector<real_type> {
    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> vec(num_features_ * (num_data_points_ - 1 + boundary));
    #pragma omp parallel for collapse(2)
    for (size_type col = 0; col < num_features_; ++col) {
        for (size_type row = 0; row < num_data_points_ - 1; ++row) {
            vec[col * (num_data_points_ - 1 + boundary) + row] = data_[row][col];
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
