#include <chrono>
#include <omp.h>
#include <plssvm/backends/OpenMP/OpenMP_CSVM.hpp>
#include <plssvm/backends/OpenMP/svm-kernel.hpp>
#include <plssvm/detail/operators.hpp>
#include <plssvm/detail/string_utility.hpp>
namespace plssvm {

OpenMP_CSVM::OpenMP_CSVM(real_t cost_,
                         real_t epsilon_,
                         kernel_type kernel_,
                         real_t degree_,
                         real_t gamma_,
                         real_t coef0_,
                         bool info_) :
    CSVM<real_t>(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}

std::vector<real_t> OpenMP_CSVM::generate_q() {
    std::vector<real_t> q;
    if (info) {
        std::cout << "kernel_q" << std::endl;
    }
    q.reserve(data.size());
    for (int i = 0; i < data.size() - 1; ++i) {
        q.emplace_back(kernel_function(data.back(), data[i]));
    }
    return q;
}

std::vector<real_t> OpenMP_CSVM::CG(const std::vector<real_t> &b, const int imax, const real_t eps, const std::vector<real_t> &q) {
    std::vector<real_t> x(b.size(), 1);
    real_t *datlast = &data.back()[0];
    static const size_t dim = data.back().size();
    static const size_t dept = b.size();

    assert(dim == num_features);
    assert(dept == num_data_points - 1);
    //r = b - (A * x)
    ///r = b;
    std::vector<real_t> r(b);

    std::vector<real_t> ones(b.size(), 1.0);

    kernel_linear(b, data, datlast, q.data(), r, ones.data(), dim, QA_cost, 1 / cost, -1);  // TODO: other kernels

    std::cout << "r= b-Ax" << std::endl;
    std::vector<real_t> d(b.size(), 0.0);

    std::memcpy(d.data(), r.data(), dept * sizeof(real_t));
    // delta \gets r^Tr
    real_t delta = mult(r.data(), r.data(), r.size());
    const real_t delta0 = delta;
    real_t alpha, beta;

    for (int run = 0; run < imax; ++run) {
        std::cout << "Start Iteration: " << run << std::endl;
        //Ad = A * d
        std::vector<real_t> Ad(dept, 0.0);

        kernel_linear(b, data, datlast, q.data(), Ad, d.data(), dim, QA_cost, 1 / cost, 1);  // TODO: other kernels

        alpha = delta / mult(d.data(), Ad.data(), d.size());
        x += mult(alpha, d.data(), d.size());
        //r = b - (A * x)
        ///r = b;
        std::copy(b.begin(), b.end(), r.begin());

        kernel_linear(b, data, datlast, q.data(), r, x.data(), dim, QA_cost, 1 / cost, -1);  // TODO: other kernels

        delta = mult(r.data(), r.data(), r.size());
        //break;
        if (delta < eps * eps * delta0)
            break;
        beta = -mult(r.data(), Ad.data(), b.size()) / mult(d.data(), Ad.data(), d.size());
        add(mult(beta, d.data(), d.size()), r.data(), d.data(), d.size());
    }

    return x;
}

}  // namespace plssvm