/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenMP/cg_explicit/blas.hpp"

#include "plssvm/constants.hpp"      // plssvm::real_type
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"         // plssvm::soa_matrix

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

void device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A.size() == m * k, "A matrix sizes mismatch!: {} != {}", A.size(), m * k);
    PLSSVM_ASSERT(B.shape() == (std::array<std::size_t, 2>{ n, k }), "B matrix sizes mismatch!: [{}] != [{}, {}]", fmt::join(B.shape(), ", "), n, k);
    PLSSVM_ASSERT(C.shape() == (std::array<std::size_t, 2>{ n, m }), "C matrix sizes mismatch!: [{}] != [{}, {}]", fmt::join(C.shape(), ", "), n, m);

    #pragma omp parallel for collapse(2) default(none) shared(A, B, C) firstprivate(n, m, k, alpha, beta)
    for (std::size_t rhs = 0; rhs < n; ++rhs) {
        for (std::size_t row = 0; row < m; ++row) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < k; ++dim) {
                temp += A[row * k + dim] * B(rhs, dim);
            }
            C(rhs, row) = alpha * temp + beta * C(rhs, row);
        }
    }
}

void device_kernel_symm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const std::vector<real_type> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A.size() == m * (k + 1) / 2, "A matrix sizes mismatch!: {} != {}", A.size(), m * (k + 1) / 2);
    PLSSVM_ASSERT(B.shape() == (std::array<std::size_t, 2>{ n, k }), "B matrix sizes mismatch!: [{}] != [{}, {}]", fmt::join(B.shape(), ", "), n, k);
    PLSSVM_ASSERT(C.shape() == (std::array<std::size_t, 2>{ n, m }), "C matrix sizes mismatch!: [{}] != [{}, {}]", fmt::join(C.shape(), ", "), n, m);

    #pragma omp parallel for collapse(2) default(none) shared(A, B, C) firstprivate(n, m, k, alpha, beta)
    for (std::size_t rhs = 0; rhs < n; ++rhs) {
        for (std::size_t row = 0; row < m; ++row) {
            real_type temp{ 0.0 };
            unsigned long long offset{ 0 };

            // left of the diagonal -> use symmetrically mirrored values
            #pragma omp simd reduction(+ : temp) reduction(+ : offset)
            for (unsigned long long dim = 0; dim < row; ++dim) {
                offset += dim;
                temp += A[dim * k + row - offset] * B(rhs, dim);
            }
            // diagonal + right of the diagonal -> use contiguous values
            offset += row;
            #pragma omp simd reduction(+ : temp)
            for (unsigned long long dim = row; dim < k; ++dim) {
                temp += A[row * k + dim - offset] * B(rhs, dim);
            }

            C(rhs, row) = alpha * temp + beta * C(rhs, row);
        }
    }
}

}  // namespace plssvm::openmp