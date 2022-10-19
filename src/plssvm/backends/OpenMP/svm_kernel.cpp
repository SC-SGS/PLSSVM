/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "plssvm/constants.hpp"              // plssvm::kernel_index_type, plssvm::OPENMP_BLOCK_SIZE
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function

#include <utility>  // std::forward
#include <vector>   // std::vector

namespace plssvm::openmp {

namespace detail {

template <kernel_function_type kernel, typename real_type, typename... Args>
void device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, Args &&...args) {
    PLSSVM_ASSERT(q.size() == data.size() - 1, "Sizes mismatch!: {} != {}", q.size(), data.size() - 1);
    PLSSVM_ASSERT(q.size() == ret.size(), "Sizes mismatch!: {} != {}", q.size(), ret.size());
    PLSSVM_ASSERT(q.size() == d.size(), "Sizes mismatch!: {} != {}", q.size(), d.size());
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");
    PLSSVM_ASSERT(add == real_type{ -1.0 } || add == real_type{ 1.0 }, "add must either be -1.0 or 1.0, but is {}!", add);

    const auto dept = static_cast<kernel_index_type>(d.size());

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (kernel_index_type i = 0; i < dept; i += OPENMP_BLOCK_SIZE) {
        for (kernel_index_type j = 0; j < dept; j += OPENMP_BLOCK_SIZE) {
            for (kernel_index_type ii = 0; ii < OPENMP_BLOCK_SIZE && ii + i < dept; ++ii) {
                real_type ret_iii = 0.0;
                for (kernel_index_type jj = 0; jj < OPENMP_BLOCK_SIZE && jj + j < dept; ++jj) {
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

}  // namespace detail

template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add) {
    detail::device_kernel<kernel_function_type::linear>(q, ret, d, data, QA_cost, cost, add);
}
template void device_kernel_linear(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, float, float, float);
template void device_kernel_linear(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, double, double, double);

template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel<kernel_function_type::polynomial>(q, ret, d, data, QA_cost, cost, add, degree, gamma, coef0);
}
template void device_kernel_poly(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, float, float, float, int, float, float);
template void device_kernel_poly(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, double, double, double, int, double, double);

template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const real_type add, const real_type gamma) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel<kernel_function_type::rbf>(q, ret, d, data, QA_cost, cost, add, gamma);
}
template void device_kernel_radial(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, float, float, float, float);
template void device_kernel_radial(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, double, double, double, double);

}  // namespace plssvm::openmp
