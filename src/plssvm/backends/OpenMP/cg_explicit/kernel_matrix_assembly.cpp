/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/cg_explicit/kernel_matrix_assembly.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type, plssvm::kernel_function
#include "plssvm/matrix.hpp"                 // plssvm::soa_matrix

#include <cstddef>  // std::size_t
#include <utility>  // std::forward
#include <vector>   // std::vector

namespace plssvm::openmp {

namespace detail {

template <kernel_function_type kernel, typename... Args>
void device_kernel_assembly(const std::vector<real_type> &q, std::vector<real_type> &ret, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost, Args &&...args) {
    PLSSVM_ASSERT(q.size() == data.num_rows() - 1, "Sizes mismatch!: {} != {}", q.size(), data.num_rows() - 1);
#if defined(PLSSVM_USE_GEMM)
    PLSSVM_ASSERT(ret.size() == q.size() * q.size(), "Sizes mismatch (GEMM)!: {} != {}", ret.size(), q.size() * q.size());
#else
    PLSSVM_ASSERT(ret.size() == q.size() * (q.size() + 1) / 2, "Sizes mismatch (SYMM)!: {} != {}", ret.size(), q.size() * (q.size() + 1) / 2);
#endif
    PLSSVM_ASSERT(cost != real_type{ 0.0 }, "cost must not be 0.0 since it is 1 / plssvm::cost!");

    const std::size_t dept = q.size();

    #pragma omp parallel for schedule(dynamic)
    for (std::size_t row = 0; row < dept; ++row) {
        for (std::size_t col = row + 1; col < dept; ++col) {
            const real_type temp = kernel_function<kernel>(data, row, data, col, std::forward<Args>(args)...) + QA_cost - q[row] - q[col];
#if defined(PLSSVM_USE_GEMM)
            ret[row * dept + col] = temp;
            ret[col * dept + row] = temp;
#else
            ret[row * dept + col - row * (row + 1) / 2] = temp;
#endif
        }
    }

    // apply cost to diagonal
    #pragma omp parallel for
    for (std::size_t i = 0; i < dept; ++i) {
        const real_type temp = kernel_function<kernel>(data, i, data, i, std::forward<Args>(args)...) + cost + QA_cost - q[i] - q[i];
#if defined(PLSSVM_USE_GEMM)
        ret[i * dept + i] = temp;
#else
        ret[i * dept + i - i * (i + 1) / 2] = temp;
#endif
    }
}

}  // namespace detail

void device_kernel_assembly_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost) {
    detail::device_kernel_assembly<kernel_function_type::linear>(q, ret, data, QA_cost, cost);
}

void device_kernel_assembly_polynomial(const std::vector<real_type> &q, std::vector<real_type> &ret, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly<kernel_function_type::polynomial>(q, ret, data, QA_cost, cost, degree, gamma, coef0);
}

void device_kernel_assembly_rbf(const std::vector<real_type> &q, std::vector<real_type> &ret, const soa_matrix<real_type> &data, const real_type QA_cost, const real_type cost, const real_type gamma) {
    PLSSVM_ASSERT(gamma > real_type{ 0.0 }, "gamma must be greater than 0, but is {}!", gamma);

    detail::device_kernel_assembly<kernel_function_type::rbf>(q, ret, data, QA_cost, cost, gamma);
}

}  // namespace plssvm::openmp
