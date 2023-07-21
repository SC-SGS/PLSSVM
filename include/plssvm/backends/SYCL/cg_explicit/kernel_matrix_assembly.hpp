/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::pow, sycl::exp

namespace plssvm::sycl::detail {

/**
 * @brief Create the explicit kernel matrix using the linear kernel function (\f$\vec{u}^T \cdot \vec{v}\f$).
 */
class device_kernel_assembly_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] ret the calculated kernel matrix
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     */
    device_kernel_assembly_linear(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost) :
        ret_{ ret }, data_d_{ data_d }, num_rows_{ num_rows }, num_features_{ num_features }, q_{ q }, QA_cost_{ QA_cost }, cost_{ cost } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0);
        const unsigned long long j = nd_idx.get_global_id(1);

        if (i < num_rows_ && j < num_rows_ && j >= i) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += data_d_[dim * (num_rows_ + 1) + i] * data_d_[dim * (num_rows_ + 1) + j];
            }
            temp = temp + QA_cost_ - q_[i] - q_[j];
            if (i == j) {
                temp += cost_;
            }

            ret_[i * num_rows_ + j - i * (i + 1) / 2] = temp;
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *ret_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type *q_;
    const real_type QA_cost_;
    const real_type cost_;
    /// @endcond
};

/**
 * @brief Create the explicit kernel matrix using the polynomial kernel function (\f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$).
 */
class device_kernel_assembly_polynomial {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] ret the calculated kernel matrix
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial kernel function
     * @param[in] coef0 parameter used in the polynomial kernel function
     */
    device_kernel_assembly_polynomial(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const real_type degree, const real_type gamma, const real_type coef0) :
        ret_{ ret }, data_d_{ data_d }, num_rows_{ num_rows }, num_features_{ num_features }, q_{ q }, QA_cost_{ QA_cost }, cost_{ cost }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0);
        const unsigned long long j = nd_idx.get_global_id(1);

        if (i < num_rows_ && j < num_rows_ && j >= i) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += data_d_[dim * (num_rows_ + 1) + i] * data_d_[dim * (num_rows_ + 1) + j];
            }
            temp = ::sycl::pow(gamma_ * temp + coef0_, degree_) + QA_cost_ - q_[i] - q_[j];
            if (i == j) {
                temp += cost_;
            }

            ret_[i * num_rows_ + j - i * (i + 1) / 2] = temp;
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *ret_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type *q_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type degree_;  // no overload for sycl::pow(double, int) -> sycl::pown(double, int)
    const real_type gamma_;
    const real_type coef0_;
    /// @endcond
};

/**
 * Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 */
class device_kernel_assembly_rbf {
    static constexpr unsigned long long WARP_SIZE = 32;
    static constexpr unsigned long long BLOCK_SIZE = 16;

  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] ret the calculated kernel matrix
     * @param[in] data_d the data points to calculate the kernel matrix from
     * @param[in] num_rows the number of data points
     * @param[in] num_features the number of features per data point
     * @param[in] q the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @param[in] cost the cost factor the diagonal is scaled with
     * @param[in] gamma parameter used in the rbf kernel function
     */
    device_kernel_assembly_rbf(::sycl::handler &cgh, real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const real_type gamma) :
        data_cache_i_{ ::sycl::range<2>{ BLOCK_SIZE, WARP_SIZE }, cgh }, data_cache_j_{ ::sycl::range<2>{ BLOCK_SIZE, WARP_SIZE }, cgh },
        ret_{ ret }, data_d_{ data_d }, num_rows_{ num_rows }, num_features_{ num_features }, q_{ q }, QA_cost_{ QA_cost }, cost_{ cost }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long i = nd_idx.get_global_id(0);
        const unsigned long long i_cached_idx = nd_idx.get_group(0) * nd_idx.get_local_range(0) + nd_idx.get_local_id(1);
        const unsigned long long j = nd_idx.get_global_id(1);

        if (nd_idx.get_group(0) >= nd_idx.get_group(1)) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; dim += BLOCK_SIZE) {
                // zero out shared memory
                if (nd_idx.get_local_id(0) < BLOCK_SIZE) {
                    data_cache_i_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
                    data_cache_j_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = real_type{ 0.0 };
                }

                // load data into shared memory
                if (nd_idx.get_local_id(0) < BLOCK_SIZE && dim + nd_idx.get_local_id(0) < num_features_) {
                    if (i_cached_idx < num_rows_) {
                        data_cache_i_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1) + i_cached_idx];
                    }
                    if (j < num_rows_) {
                        data_cache_j_[nd_idx.get_local_id(0)][nd_idx.get_local_id(1)] = data_d_[(dim + nd_idx.get_local_id(0)) * (num_rows_ + 1) + j];
                    }
                }
                nd_idx.barrier();

                // calculation
                for (unsigned long long block_dim = 0; block_dim < BLOCK_SIZE; ++block_dim) {
                    const real_type d = data_cache_i_[block_dim][nd_idx.get_local_id(0)] - data_cache_j_[block_dim][nd_idx.get_local_id(1)];
                    temp += d * d;
                }
                nd_idx.barrier();
            }

            if (i < num_rows_ && j < num_rows_ && i >= j) {
                temp = ::sycl::exp(-gamma_ * temp) + QA_cost_ - q_[i] - q_[j];
                if (i == j) {
                    temp += cost_;
                }

                ret_[j * num_rows_ + i - j * (j + 1) / 2] = temp;
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_i_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_j_;

    /// @cond Doxygen_suppress
    real_type *ret_;
    const real_type *data_d_;
    const unsigned long long num_rows_;
    const unsigned long long num_features_;
    const real_type *q_;
    const real_type QA_cost_;
    const real_type cost_;
    const real_type gamma_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HPP_
