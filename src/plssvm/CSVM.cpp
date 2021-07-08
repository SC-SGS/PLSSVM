#include <plssvm/CSVM.hpp>

#include "fmt/core.h"  // fmt::print

namespace plssvm {

template <typename T>
void CSVM<T>::learn() {
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

    if (print_info_) {
        fmt::print("Start CG\n");
    }

    // solve minimization
    alpha_ = CG(b, num_features_, epsilon_, q);
    alpha_.emplace_back(-sum(alpha_));
    bias_ = value_.back() - QA_cost_ * alpha_.back() - (q * alpha_);
}

template <typename T>
auto CSVM<T>::kernel_function(const real_type *xi, const real_type *xj, const size_type dim) -> real_type {  // TODO: const correctness
    switch (kernel_) {
        case kernel_type::linear:
            return mult(xi, xj, dim);
        case kernel_type::polynomial:
            return std::pow(gamma_ * mult(xi, xj, dim) + coef0_, degree_);
        case kernel_type::rbf: {
            real_type temp = 0;
            for (int i = 0; i < dim; ++i) {
                temp += (xi[i] - xj[i]);
            }
            return exp(-gamma_ * temp * temp);
        }
        default:
            throw std::runtime_error("Can not decide which kernel!");
    }
}

template <typename T>
auto CSVM<T>::kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj) -> real_type {
    // TODO: check sizes?
    return kernel_function(xi.data(), xj.data(), xi.size());
}

template <typename T>
void CSVM<T>::learn(const std::string &input_filename, const std::string &model_filename) {
    // parse data file
    parse_file(input_filename);

    // load all necessary data onto the device
    loadDataDevice();

    // learn model
    learn();

    // write results to model file
    write_model(model_filename);

    //    if (true) {  // TODO: check
    //        std::clog << data.size() << ", " << num_features << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << std::endl;
    //    }
}

// explicitly instantiate template class
template class CSVM<float>;
template class CSVM<double>;

}  // namespace plssvm