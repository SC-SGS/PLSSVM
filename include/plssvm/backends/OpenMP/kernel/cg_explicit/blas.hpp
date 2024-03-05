/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the CUDA backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_

#include "plssvm/constants.hpp"      // plssvm::real_type
#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT
#include "plssvm/matrix.hpp"         // plssvm::aos_matrix
#include "plssvm/shape.hpp"          // plssvm::shape

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

/**
 * @brief Perform an explicit BLAS SYMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` symmetric matrix (memory optimized), @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
 * @param[in] m the number of rows in @p A and @p C
 * @param[in] n the number of columns in @p B and @p C
 * @param[in] k the number of rows in @p A and number of columns in @p B
 * @param[in] alpha the scalar alpha value
 * @param[in] A the matrix @p A
 * @param[in] B the matrix @p B
 * @param[in] beta the scalar beta value
 * @param[in,out] C the matrix @p C, also used as result matrix
 */
inline void device_kernel_symm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const std::vector<real_type> &A, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) {
    // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
    PLSSVM_ASSERT(A.size() == (m + PADDING_SIZE) * (k + PADDING_SIZE + 1) / 2, "A matrix sizes mismatch!: {} != {}", A.size(), (m + PADDING_SIZE) * (k + PADDING_SIZE + 1) / 2);
    PLSSVM_ASSERT(B.shape() == (plssvm::shape{ n, k }), "B matrix sizes mismatch!: {} != [{}, {}]", B.shape(), n, k);
    PLSSVM_ASSERT(C.shape() == (plssvm::shape{ n, m }), "C matrix sizes mismatch!: {} != [{}, {}]", C.shape(), n, m);

#pragma omp parallel for collapse(2) default(none) shared(A, B, C) firstprivate(n, m, k, alpha, beta)
    for (std::size_t rhs = 0; rhs < n; ++rhs) {
        for (std::size_t row = 0; row < m; ++row) {
            real_type temp{ 0.0 };
            unsigned long long offset{ 0 };

            // left of the diagonal -> use symmetrically mirrored values
            for (unsigned long long dim = 0; dim < row; ++dim) {
                offset += dim;
                temp += A[dim * (k + PADDING_SIZE) + row - offset] * B(rhs, dim);
            }
            // diagonal + right of the diagonal -> use contiguous values
            offset += row;
#pragma omp simd reduction(+ : temp)
            for (unsigned long long dim = row; dim < k; ++dim) {
                temp += A[row * (k + PADDING_SIZE) + dim - offset] * B(rhs, dim);
            }

            C(rhs, row) = alpha * temp + beta * C(rhs, row);
        }
    }
}

}  // namespace plssvm::openmp

#endif  // PLSSVM_BACKENDS_OPENMP_KERNEL_CG_EXPLICIT_BLAS_HPP_