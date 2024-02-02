/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for implicitly assembling the kernel matrix using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
#include "plssvm/constants.hpp"                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::pown, sycl::exp

namespace plssvm::sycl::detail {

/**
 * @brief Create the explicit kernel matrix using the linear kernel function (\f$\vec{u}^T \cdot \vec{v}\f$).
 */
class device_kernel_assembly_linear_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] B the matrix @p B
     * @param[in,out] C the matrix @p C
     * @param[in] num_classes the number of classes in the data set
     */
    device_kernel_assembly_linear_symm(::sycl::handler &cgh, const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type *B, real_type *C, const unsigned long long num_classes) :
        data_cache_i_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        data_cache_j_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        alpha_{ alpha },
        q_{ q },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        if (nd_idx.get_group(0) >= nd_idx.get_group(1)) {
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
            for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = i_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                    const unsigned long long global_j = j_linear + internal * THREAD_BLOCK_SIZE;

                    data_cache_i_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_i_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_j_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    data_cache_j_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                }
                nd_idx.barrier();

                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += data_cache_i_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                }
                nd_idx.barrier();
            }

            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = i + internal_i;
                    const unsigned long long global_j = j + internal_j;

                    if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = temp_ij + QA_cost_ - q_[global_i] - q_[global_j];
                        if (global_i == global_j) {
                            temp_ij += cost_;
                        }

                        // apply B and C
                        for (unsigned long long class_idx = 0; class_idx < num_classes_; ++class_idx) {
                            detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_j * (num_classes_ + PADDING_SIZE) + class_idx];
                            if (global_i != global_j) {
                                detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_i * (num_classes_ + PADDING_SIZE) + class_idx];
                            }
                        }
                    }
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_j_;

    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type *B_;
    real_type *C_;
    const unsigned long long num_classes_;
    /// @endcond
};

/**
 * @brief Create the explicit kernel matrix using the polynomial kernel function (\f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$).
 */
class device_kernel_assembly_polynomial_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial kernel function
     * @param[in] coef0 parameter used in the polynomial kernel function
     * @param[in] B the matrix @p B
     * @param[in,out] C the matrix @p C
     * @param[in] num_classes the number of classes in the data set
     */
    device_kernel_assembly_polynomial_symm(::sycl::handler &cgh, const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0, const real_type *B, real_type *C, const unsigned long long num_classes) :
        data_cache_i_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        data_cache_j_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        alpha_{ alpha },
        q_{ q },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        degree_{ degree },
        gamma_{ gamma },
        coef0_{ coef0 },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        if (nd_idx.get_group(0) >= nd_idx.get_group(1)) {
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
            for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = i_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                    const unsigned long long global_j = j_linear + internal * THREAD_BLOCK_SIZE;

                    data_cache_i_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_i_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_j_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    data_cache_j_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                }
                nd_idx.barrier();

                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            temp[internal_i][internal_j] += data_cache_i_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                        }
                    }
                }
                nd_idx.barrier();
            }

            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = i + internal_i;
                    const unsigned long long global_j = j + internal_j;

                    if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = ::sycl::pown(gamma_ * temp_ij + coef0_, degree_) + QA_cost_ - q_[global_i] - q_[global_j];
                        if (global_i == global_j) {
                            temp_ij += cost_;
                        }

                        // apply B and C
                        for (unsigned long long class_idx = 0; class_idx < num_classes_; ++class_idx) {
                            detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_j * (num_classes_ + PADDING_SIZE) + class_idx];
                            if (global_i != global_j) {
                                detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_i * (num_classes_ + PADDING_SIZE) + class_idx];
                            }
                        }
                    }
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_j_;

    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type QA_cost_;
    const real_type cost_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
    const real_type *B_;
    real_type *C_;
    const unsigned long long num_classes_;
    /// @endcond
};

/**
 * Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 */
class device_kernel_assembly_rbf_symm {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in] alpha the scalar alpha value
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] gamma parameter used in the polynomial kernel function
     * @param[in] B the matrix @p B
     * @param[in,out] C the matrix @p C
     * @param[in] num_classes the number of classes in the data set
     */
    device_kernel_assembly_rbf_symm(::sycl::handler &cgh, const real_type alpha, const real_type *q, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type QA_cost, const real_type cost, const real_type gamma, const real_type *B, real_type *C, const unsigned long long num_classes) :
        data_cache_i_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        data_cache_j_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh },
        alpha_{ alpha },
        q_{ q },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        QA_cost_{ QA_cost },
        cost_{ cost },
        gamma_{ gamma },
        B_{ B },
        C_{ C },
        num_classes_{ num_classes } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long i_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long j_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        if (nd_idx.get_group(0) >= nd_idx.get_group(1)) {
            real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };
            for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
                // load data into shared memory
                for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                    const unsigned long long global_i = i_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                    const unsigned long long global_j = j_linear + internal * THREAD_BLOCK_SIZE;

                    data_cache_i_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_i_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                    data_cache_j_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    data_cache_j_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                }
                nd_idx.barrier();

                // calculation
                for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                    for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                        for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                            const real_type d = data_cache_i_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] - data_cache_j_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                            temp[internal_i][internal_j] += d * d;
                        }
                    }
                }
                nd_idx.barrier();
            }

            for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                    const unsigned long long global_i = i + internal_i;
                    const unsigned long long global_j = j + internal_j;

                    if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                        real_type temp_ij = temp[internal_i][internal_j];
                        temp_ij = ::sycl::exp(-gamma_ * temp_ij) + QA_cost_ - q_[global_i] - q_[global_j];
                        if (global_i == global_j) {
                            temp_ij += cost_;
                        }

                        // apply B and C
                        for (unsigned long long class_idx = 0; class_idx < num_classes_; ++class_idx) {
                            detail::atomic_op<real_type>{ C_[global_i * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_j * (num_classes_ + PADDING_SIZE) + class_idx];
                            if (global_i != global_j) {
                                detail::atomic_op<real_type>{ C_[global_j * (num_classes_ + PADDING_SIZE) + class_idx] } += alpha_ * temp_ij * B_[global_i * (num_classes_ + PADDING_SIZE) + class_idx];
                            }
                        }
                    }
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_j_;

    /// @cond Doxygen_suppress
    const real_type alpha_;
    const real_type *q_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type gamma_;
    const real_type *B_;
    real_type *C_;
    const unsigned long long num_classes_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_IMPLICIT_KERNEL_MATRIX_ASSEMBLY_BLAS_HPP_
