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

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::print

#include <chrono>  // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <memory>  // std::make_shared
#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    target_{ params.target }, kernel_{ params.kernel }, degree_{ params.degree }, gamma_{ params.gamma }, coef0_{ params.coef0 }, cost_{ params.cost }, epsilon_{ params.epsilon }, print_info_{ params.print_info }, data_ptr_{ params.data_ptr }, value_ptr_{ params.value_ptr }, alpha_ptr_{ params.alphas_ptr }, bias_{ -params.rho } {
    if (data_ptr_ == nullptr) {
        throw exception{ "No data points provided!" };
    }

    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");

    num_data_points_ = data_ptr_->size();
    num_features_ = (*data_ptr_)[0].size();
}

template <typename T>
void csvm<T>::learn() {
    using namespace plssvm::operators;

    if (value_ptr_ == nullptr) {
        throw exception{ "No labels provided for training!" };
    }

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");
    PLSSVM_ASSERT(data_ptr_->size() == value_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), value_ptr_->size());

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
void csvm<T>::update_w() {
    w_.resize(num_features_);
    std::fill(w_.begin(), w_.end(), 0.0);
    for (size_type data_index = 0; data_index < num_data_points_; ++data_index) {
        // w_ += (*alpha_ptr_)[data_index] * (*data_ptr_)[data_index];
        for (size_type feature_index = 0; feature_index < num_features_; ++feature_index) {
            w_[feature_index] += (*alpha_ptr_)[data_index] * (*data_ptr_)[data_index][feature_index];
        }
    }
}

template <typename T>
auto csvm<T>::predict(const std::vector<real_type> &point) -> real_type {
    using namespace plssvm::operators;

    if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas provided for prediction!" };
    }

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");
    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());
    PLSSVM_ASSERT((*data_ptr_)[0].size() == point.size(), "Prediction point has different amount of features than training data!");

    real_type temp = bias_;

    if (kernel_ == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        if (w_.empty()) {
            update_w();
        }
        temp += transposed{ w_ } * point;
    } else {
        for (size_type data_index = 0; data_index < num_data_points_; ++data_index) {
            temp += (*alpha_ptr_)[data_index] * kernel_function((*data_ptr_)[data_index], point);
        }
    }

    return temp;
}

template <typename T>
auto csvm<T>::predict_label(const std::vector<real_type> &point) -> real_type {
    using namespace plssvm::operators;
    return static_cast<real_type>(sign(predict(point)));
}

template <typename T>
auto csvm<T>::predict(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    std::vector<real_type> classes;
    classes.reserve(points.size());
    for (const std::vector<real_type> &point : points) {
        classes.emplace_back(predict(point));
    }
    return classes;
}

template <typename T>
auto csvm<T>::predict_label(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    using namespace plssvm::operators;
    std::vector<real_type> classes(predict(points));
    for (real_type &elem : classes) {
        elem = sign(elem);
    }
    return classes;
}

template <typename T>
auto csvm<T>::accuracy() -> real_type {
    using namespace plssvm::operators;

    if (value_ptr_ == nullptr) {
        throw exception{ "No labels provided for accuracy calculation!" };
    }

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");
    PLSSVM_ASSERT(data_ptr_->size() == value_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());

    int correct = 0;
    std::vector<real_type> predictions = predict(*data_ptr_);
    for (size_type index = 0; index < predictions.size(); ++index) {
        if (predictions[index] * (*value_ptr_)[index] > real_type{ 0.0 }) {
            ++correct;
        }
    }
    return static_cast<real_type>(correct) / static_cast<real_type>(num_data_points_);
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
    }
    throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(kernel_)) };
}

template <typename T>
auto csvm<T>::transform_data(const std::vector<std::vector<real_type>> &matrix, const size_type boundary, const size_type num_points) -> std::vector<real_type> {
    PLSSVM_ASSERT(!matrix.empty(), "Matrix is empty!");
    PLSSVM_ASSERT(num_points <= matrix.size(), "Num points to transform can not exceed matrix size!");

    const size_type num_features = matrix[0].size();

    for (const std::vector<real_type> &point : matrix) {
        PLSSVM_ASSERT(point.size() == num_features_, "Feature sizes mismatch!: {} != {}", point.size(), num_features_);
    }
    auto start_time = std::chrono::steady_clock::now();

    std::vector<real_type> vec(num_features * (num_points + boundary));
#pragma omp parallel for collapse(2)
    for (size_type col = 0; col < num_features; ++col) {
        for (size_type row = 0; row < num_points; ++row) {
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
