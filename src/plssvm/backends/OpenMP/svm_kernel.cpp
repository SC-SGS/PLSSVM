/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type, plssvm::kernel_function

#include <cstddef>  // std::size_t
#include <utility>  // std::forward
#include <vector>   // std::vector

namespace plssvm::openmp {

template <kernel_type kernel, typename real_type, typename... Args>
void device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, Args &&...args) {
    using size_type = std::size_t;

    constexpr size_type BLOCK_SIZE = 64;

    const size_type dept = d.size();

    #pragma omp parallel for collapse(2)
    for (size_type i = 0; i < dept; i += BLOCK_SIZE) {
        for (size_type j = 0; j < dept; j += BLOCK_SIZE) {
            for (size_type ii = 0; ii < BLOCK_SIZE && ii + i < dept; ++ii) {
                real_type ret_iii = 0.0;
                for (size_type jj = 0; jj < BLOCK_SIZE && jj + j < dept; ++jj) {
                    if (ii + i >= jj + j) {
                        const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                        if (ii + i == jj + j) {
                            ret_iii += (temp + cost * add) * d[ii + i];
                        } else {
                            ret_iii += temp * d[jj + j];
                            #pragma omp atomic
                            ret[jj + j] += temp * d[ii + i];
                        }
                    }
                }
                #pragma omp atomic
                ret[ii + i] += ret_iii;
            }
        }
    }
}

template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add) {
    device_kernel<kernel_type::linear>(q, ret, d, data, QA_cost, cost, add);
}
template void device_kernel_linear(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float);
template void device_kernel_linear(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double);

template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    device_kernel<kernel_type::polynomial>(q, ret, d, data, QA_cost, cost, add, degree, gamma, coef0);
}
template void device_kernel_poly(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const int, const float, const float);
template void device_kernel_poly(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const int, const double, const double);

template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const real_type gamma) {
    device_kernel<kernel_type::rbf>(q, ret, d, data, QA_cost, cost, add, gamma);
}
template void device_kernel_radial(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const float, const float);
template void device_kernel_radial(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const double, const double);

}  // namespace plssvm::openmp
