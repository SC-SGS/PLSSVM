/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/q_kernel.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix

#include <cstddef>                           // std::size_t
#include <vector>                            // std::vector

namespace plssvm::openmp {

template <typename real_type>
void device_kernel_q_linear(std::vector<real_type> &q, const aos_matrix<real_type> &data) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);

    #pragma omp parallel for default(none) shared(q, data)
    for (std::size_t i = 0; i < data.num_rows() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::linear>(data, i, data, data.num_rows() - 1);
    }
}
template void device_kernel_q_linear(std::vector<float> &, const aos_matrix<float> &);
template void device_kernel_q_linear(std::vector<double> &, const aos_matrix<double> &);

template <typename real_type>
void device_kernel_q_polynomial(std::vector<real_type> &q, const aos_matrix<real_type> &data, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    #pragma omp parallel for default(none) shared(q, data) firstprivate(degree, gamma, coef0)
    for (std::size_t i = 0; i < data.num_rows() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::polynomial>(data, i, data, data.num_rows() - 1, degree, gamma, coef0);
    }
}
template void device_kernel_q_polynomial(std::vector<float> &, const aos_matrix<float> &, int, float, float);
template void device_kernel_q_polynomial(std::vector<double> &, const aos_matrix<double> &, int, double, double);

template <typename real_type>
void device_kernel_q_rbf(std::vector<real_type> &q, const aos_matrix<real_type> &data, const real_type gamma) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    #pragma omp parallel for default(none) shared(q, data) firstprivate(gamma)
    for (std::size_t i = 0; i < data.num_rows() - 1; ++i) {
        q[i] = kernel_function<kernel_function_type::rbf>(data, i, data, data.num_rows() - 1, gamma);
    }
}
template void device_kernel_q_rbf(std::vector<float> &, const aos_matrix<float> &, float);
template void device_kernel_q_rbf(std::vector<double> &, const aos_matrix<double> &, double);

}  // namespace plssvm::openmp