/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/q_kernel.hpp"

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function

#include <vector>  // std::vector

namespace plssvm::openmp {

template <typename real_type>
void device_kernel_q_linear(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data) {
    PLSSVM_ASSERT(q.size() == data.size() - 1, "Sizes mismatch!: {} != {}", q.size(), data.size() - 1);

    #pragma omp parallel for
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::linear>(data[i], data.back());
    }
}
template void device_kernel_q_linear(std::vector<float> &, const std::vector<std::vector<float>> &);
template void device_kernel_q_linear(std::vector<double> &, const std::vector<std::vector<double>> &);

template <typename real_type>
void device_kernel_q_poly(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(q.size() == data.size() - 1, "Sizes mismatch!: {} != {}", q.size(), data.size() - 1);

    #pragma omp parallel for
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::polynomial>(data[i], data.back(), degree, gamma, coef0);
    }
}
template void device_kernel_q_poly(std::vector<float> &, const std::vector<std::vector<float>> &, const int, const float, const float);
template void device_kernel_q_poly(std::vector<double> &, const std::vector<std::vector<double>> &, const int, const double, const double);

template <typename real_type>
void device_kernel_q_radial(std::vector<real_type> &q, const std::vector<std::vector<real_type>> &data, const real_type gamma) {
    PLSSVM_ASSERT(q.size() == data.size() - 1, "Sizes mismatch!: {} != {}", q.size(), data.size() - 1);

    #pragma omp parallel for
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::rbf>(data[i], data.back(), gamma);
    }
}
template void device_kernel_q_radial(std::vector<float> &, const std::vector<std::vector<float>> &, const float);
template void device_kernel_q_radial(std::vector<double> &, const std::vector<std::vector<double>> &, const double);

}  // namespace plssvm::openmp