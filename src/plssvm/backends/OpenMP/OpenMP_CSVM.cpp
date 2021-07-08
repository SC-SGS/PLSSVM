#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"

#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/detail/operators.hpp"
#include "plssvm/detail/string_utility.hpp"

#include <vector>

namespace plssvm {

template <typename T>
OpenMP_CSVM<T>::OpenMP_CSVM(real_type cost,
                            real_type epsilon,
                            kernel_type kernel,
                            real_type degree,
                            real_type gamma,
                            real_type coef0,
                            bool info) :
    CSVM<T>(cost, epsilon, kernel, degree, gamma, coef0, info) {}

template <typename T>
auto OpenMP_CSVM<T>::generate_q() -> std::vector<real_type> {
    std::vector<real_type> q;
    if (print_info_) {
        std::cout << "kernel_q" << std::endl;
    }
    q.reserve(data_.size());
    for (size_type i = 0; i < data_.size() - 1; ++i) {
        q.emplace_back(base_type::kernel_function(data_.back(), data_[i]));
    }
    return q;
}

template <typename T>
auto OpenMP_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    std::vector<real_type> x(b.size(), 1);
    real_type *datlast = &data_.back()[0];
    static const size_type dim = data_.back().size();
    static const size_type dept = b.size();

    assert(dim == num_features);
    assert(dept == num_data_points - 1);
    //r = b - (A * x)
    ///r = b;
    std::vector<real_type> r(b);

    std::vector<real_type> ones(b.size(), 1.0);

    kernel_linear(b, data_, datlast, q.data(), r, ones.data(), dim, QA_cost_, 1 / cost_, -1);  // TODO: other kernels

    std::cout << "r= b-Ax" << std::endl;
    std::vector<real_type> d(b.size(), 0.0);

    std::memcpy(d.data(), r.data(), dept * sizeof(real_type));
    // delta \gets r^Tr
    real_type delta = mult(r.data(), r.data(), r.size());
    const real_type delta0 = delta;
    real_type alpha, beta;

    for (size_type run = 0; run < imax; ++run) {
        std::cout << "Start Iteration: " << run << std::endl;
        //Ad = A * d
        std::vector<real_type> Ad(dept, 0.0);

        kernel_linear(b, data_, datlast, q.data(), Ad, d.data(), dim, QA_cost_, 1 / cost_, 1);  // TODO: other kernels

        alpha = delta / mult(d.data(), Ad.data(), d.size());
        x += mult(alpha, d.data(), d.size());
        //r = b - (A * x)
        ///r = b;
        std::copy(b.begin(), b.end(), r.begin());

        kernel_linear(b, data_, datlast, q.data(), r, x.data(), dim, QA_cost_, 1 / cost_, -1);  // TODO: other kernels

        delta = mult(r.data(), r.data(), r.size());
        //break;
        if (delta < eps * eps * delta0)
            break;
        beta = -mult(r.data(), Ad.data(), b.size()) / mult(d.data(), Ad.data(), d.size());
        add(mult(beta, d.data(), d.size()), r.data(), d.data(), d.size());
    }

    return x;
}

// explicitly instantiate template class
template class OpenMP_CSVM<float>;
template class OpenMP_CSVM<double>;

}  // namespace plssvm