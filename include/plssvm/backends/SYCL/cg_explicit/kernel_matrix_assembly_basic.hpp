/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the SYCL backend basic data parallel kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_BASIC_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_BASIC_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::pown, sycl::exp
#include <sycl/ext/intel/fpga_extensions.hpp> // TODO: guard


namespace plssvm::sycl::detail::basic {

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
        ret_{ ret },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const unsigned long long i = idx.get_id(0);
        const unsigned long long j = idx.get_id(1);

        if (i < num_rows_ && j < num_rows_ && j >= i) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + i] * data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + j];
            }
            temp = temp + QA_cost_ - q_[i] - q_[j];
            if (i == j) {
                temp += cost_;
            }

#if defined(PLSSVM_USE_GEMM)
            ret_[i * (num_rows_ + PADDING_SIZE) + j] = temp;
            ret_[j * (num_rows_ + PADDING_SIZE) + i] = temp;
#else
            ret_[i * (num_rows_ + PADDING_SIZE) + j - i * (i + 1) / 2] = temp;
#endif
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
    device_kernel_assembly_polynomial(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) :
        ret_{ ret },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost },
        degree_{ degree },
        gamma_{ gamma },
        coef0_{ coef0 } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const unsigned long long i = idx.get_id(0);
        const unsigned long long j = idx.get_id(1);

        if (i < num_rows_ && j < num_rows_ && j >= i) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + i] * data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + j];
            }
            temp = ::sycl::pown(gamma_ * temp + coef0_, degree_) + QA_cost_ - q_[i] - q_[j];
            if (i == j) {
                temp += cost_;
            }

#if defined(PLSSVM_USE_GEMM)
            ret_[i * (num_rows_ + PADDING_SIZE) + j] = temp;
            ret_[j * (num_rows_ + PADDING_SIZE) + i] = temp;
#else
            ret_[i * (num_rows_ + PADDING_SIZE) + j - i * (i + 1) / 2] = temp;
#endif
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
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
    /// @endcond
};

/**
 * Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 */
class device_kernel_assembly_rbf {
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
    device_kernel_assembly_rbf(real_type *ret, const real_type *data_d, const unsigned long long num_rows, const unsigned long long num_features, const real_type *q, const real_type QA_cost, const real_type cost, const real_type gamma) :
        ret_{ ret },
        data_d_{ data_d },
        num_rows_{ num_rows },
        num_features_{ num_features },
        q_{ q },
        QA_cost_{ QA_cost },
        cost_{ cost },
        gamma_{ gamma } { }

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const unsigned long long i = idx.get_id(0);
        const unsigned long long j = idx.get_id(1);

        if (i < num_rows_ && j < num_rows_ && j >= i) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                const real_type d = data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + i] - data_d_[dim * (num_rows_ + 1 + PADDING_SIZE) + j];
                temp += d * d;
            }
            temp = ::sycl::exp(-gamma_ * temp) + QA_cost_ - q_[i] - q_[j];
            if (i == j) {
                temp += cost_;
            }

#if defined(PLSSVM_USE_GEMM)
            ret_[i * (num_rows_ + PADDING_SIZE) + j] = temp;
            ret_[j * (num_rows_ + PADDING_SIZE) + i] = temp;
#else
            ret_[i * (num_rows_ + PADDING_SIZE) + j - i * (i + 1) / 2] = temp;
#endif
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
    const real_type gamma_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail::basic

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_WORK_GROUP_HPP_
