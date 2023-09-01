/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the functions used for prediction for the C-SVM using the SYCL backend.
 */

#ifndef PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_
#define PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_
#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::detail::atomic_op
#include "plssvm/constants.hpp"                     // plssvm::real_type

#include "sycl/sycl.hpp"  // sycl::item, sycl::pown, sycl::exp

namespace plssvm::sycl::detail {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 */
class device_kernel_w_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in,out] w_d the vector to speedup the linear prediction
     * @param[in] alpha_d the previously learned weights
     * @param[in] sv_d the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_features the number of features per support vector
     */
    device_kernel_w_linear(real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_features) :
        w_d_{ w_d }, alpha_d_{ alpha_d }, sv_d_{ sv_d }, num_classes_{ num_classes }, num_sv_{ num_sv }, num_features_{ num_features } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const unsigned long long feature_idx = idx.get_id(0);
        const unsigned long long class_idx = idx.get_id(1);

        if (feature_idx < num_features_ && class_idx < num_classes_) {
            real_type temp{ 0.0 };
            for (unsigned long long sv = 0; sv < num_sv_; ++sv) {
                temp += alpha_d_[class_idx * num_sv_ + sv] * sv_d_[feature_idx * num_sv_ + sv];
            }
            w_d_[class_idx * num_features_ + feature_idx] = temp;
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *w_d_;
    const real_type *alpha_d_;
    const real_type *sv_d_;
    const unsigned long long num_classes_;
    const unsigned long long num_sv_;
    const unsigned long long num_features_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 */
class device_kernel_predict_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[out] out_d the predicted values
     * @param[in] w_d the vector to speedup the calculations
     * @param[in] rho_d the previously learned bias
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     */
    device_kernel_predict_linear(real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features) :
        out_d_{ out_d }, w_d_{ w_d }, rho_d_{ rho_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_predict_points_{ num_predict_points }, num_features_{ num_features } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<2> idx) const {
        const unsigned long long predict_points_idx = idx.get_id(0);
        const unsigned long long class_idx = idx.get_id(1);

        if (predict_points_idx < num_predict_points_ && class_idx < num_classes_) {
            real_type temp{ 0.0 };
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += w_d_[class_idx * num_features_ + dim] * predict_points_d_[dim * num_predict_points_ + predict_points_idx];
            }
            out_d_[predict_points_idx * num_classes_ + class_idx] = temp - rho_d_[class_idx];
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *out_d_;
    const real_type *w_d_;
    const real_type *rho_d_;
    const real_type *predict_points_d_;
    const unsigned long long num_classes_;
    const unsigned long long num_predict_points_;
    const unsigned long long num_features_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the polynomial.
 */
class device_kernel_predict_polynomial {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] out_d the predicted values
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously learned biases
     * @param[in] sv_d the support vectors
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] degree the parameter in the polynomial kernel function
     * @param[in] gamma the parameter in the polynomial kernel function
     * @param[in] coef0 the parameter in the polynomial kernel function
     */
    device_kernel_predict_polynomial(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const int degree, const real_type gamma, const real_type coef0) :
        out_d_{ out_d }, alpha_d_{ alpha_d }, rho_d_{ rho_d }, sv_d_{ sv_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_sv_{ num_sv }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<3> idx) const {
        const unsigned long long sv_idx = idx.get_id(0);
        const unsigned long long predict_points_idx = idx.get_id(1);
        const unsigned long long class_idx = idx.get_id(2);

        if (sv_idx < num_sv_ && predict_points_idx < num_predict_points_ && class_idx < num_classes_) {
            real_type temp{ 0.0 };
            // perform dot product
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                temp += sv_d_[dim * num_sv_ + sv_idx] * predict_points_d_[dim * num_predict_points_ + predict_points_idx];
            }

            // apply degree, gamma, and coef0, alpha and rho
            temp = alpha_d_[class_idx * num_sv_ + sv_idx] * ::sycl::pown(gamma_ * temp + coef0_, degree_);
            if (sv_idx == 0) {
                temp -= rho_d_[class_idx];
            }

            detail::atomic_op<real_type>{ out_d_[predict_points_idx * num_classes_ + class_idx] } += temp;
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *out_d_;
    const real_type *alpha_d_;
    const real_type *rho_d_;
    const real_type *sv_d_;
    const real_type *predict_points_d_;
    const unsigned long long num_classes_;
    const unsigned long long num_sv_;
    const unsigned long long num_predict_points_;
    const unsigned long long num_features_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the rbf.
 */
class device_kernel_predict_rbf {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] out_d the predicted values
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously learned biases
     * @param[in] sv_d the support vectors
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     * @param[in] gamma the parameter in the rbf kernel function
     */
    device_kernel_predict_rbf(real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const real_type gamma) :
        out_d_{ out_d }, alpha_d_{ alpha_d }, rho_d_{ rho_d }, sv_d_{ sv_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_sv_{ num_sv }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] idx indices representing the current point in the execution space
     */
    void operator()(::sycl::item<3> idx) const {
        const unsigned long long sv_idx = idx.get_id(0);
        const unsigned long long predict_points_idx = idx.get_id(1);
        const unsigned long long class_idx = idx.get_id(2);

        if (sv_idx < num_sv_ && predict_points_idx < num_predict_points_ && class_idx < num_classes_) {
            real_type temp{ 0.0 };
            // perform dist calculation
            for (unsigned long long dim = 0; dim < num_features_; ++dim) {
                const real_type diff = sv_d_[dim * num_sv_ + sv_idx] - predict_points_d_[dim * num_predict_points_ + predict_points_idx];
                temp += diff * diff;
            }

            // apply degree, gamma, and coef0, alpha and rho
            temp = alpha_d_[class_idx * num_sv_ + sv_idx] * ::sycl::exp(-gamma_ * temp);
            if (sv_idx == 0) {
                temp -= rho_d_[class_idx];
            }

            detail::atomic_op<real_type>{ out_d_[predict_points_idx * num_classes_ + class_idx] } += temp;
        }
    }

  private:
    /// @cond Doxygen_suppress
    real_type *out_d_;
    const real_type *alpha_d_;
    const real_type *rho_d_;
    const real_type *sv_d_;
    const real_type *predict_points_d_;
    const unsigned long long num_classes_;
    const unsigned long long num_sv_;
    const unsigned long long num_predict_points_;
    const unsigned long long num_features_;
    const real_type gamma_;
    /// @endcond
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_BACKENDS_SYCL_PREDICT_KERNEL_HPP_