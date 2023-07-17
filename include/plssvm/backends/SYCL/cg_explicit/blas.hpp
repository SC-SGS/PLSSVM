/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly performing a BLAS GEMM like matrix-matrix multiplication using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "sycl/sycl.hpp"  // sycl::nd_item

namespace plssvm::sycl::detail {

/**
 * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where A is a `m x k` matrix, B is a `k x n` matrix, C is a `m x n` matrix, and alpha and beta are scalars.
 */
class device_kernel_gemm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    device_kernel_gemm(const unsigned long long m, const unsigned long long n, const unsigned long long k, const real_type alpha, const real_type *A, const real_type *B, const real_type beta, real_type *C) :
        m_{ m }, n_{ n }, k_{ k }, alpha_{ alpha }, A_{ A }, B_{ B }, beta_{ beta }, C_{ C } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
//    void operator()(::sycl::nd_item<2> nd_idx) const {
    void operator()(::sycl::item<2> idx) const {
        // compute: C = alpha * A * B + beta * C with A in m x k, B in n x k, and C in n x m, alpha, beta as scalar
//        const unsigned long long i = nd_idx.get_global_id(0);
//        const unsigned long long j = nd_idx.get_global_id(1);
        const unsigned long long i = idx.get_id(0);
        const unsigned long long j = idx.get_id(1);

        if (i < m_ && j < n_) {
            real_type temp{ 0.0 };
            unsigned long long offset = 0;
            // left of the diagonal -> use symmetrically mirrored values
            for (unsigned long long dim = 0; dim < i; ++dim) {
                offset += dim;
                temp += A_[dim * k_ + i - offset] * B_[j * k_ + dim];
            }
            // diagonal + right of the diagonal -> use contiguous values
            offset += i;
            for (unsigned long long dim = i; dim < k_; ++dim) {
                temp += A_[i * k_ + dim - offset] * B_[j * k_ + dim];
            }
            C_[j * m_ + i] = alpha_ * temp + beta_ * C_[j * m_ + i];
        }
    }

  private:
    /// @cond Doxygen_suppress
    const unsigned long long m_;
    const unsigned long long n_;
    const unsigned long long k_;
    const real_type alpha_;
    const real_type *A_;
    const real_type *B_;
    const real_type beta_;
    real_type *C_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_BLAS_HPP_
