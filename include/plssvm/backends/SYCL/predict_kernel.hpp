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

#pragma once

#include "plssvm/backends/SYCL/detail/atomics.hpp"  // plssvm::sycl::atomic_op
#include "plssvm/constants.hpp"                     // plssvm::kernel_index_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::range, sycl::pow, sycl::exp

namespace plssvm::sycl {

/**
 * @brief Calculate the `w` vector to speed up the prediction of the labels for data points using the linear kernel function.
 * @details Supports multi-GPU execution.
 * @tparam T the type of the data
 */
template <typename T>
class device_kernel_w_linear {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel generating the `w` vector used to speedup the prediction when using the linear kernel function.
     * @details Currently only single GPU execution is supported.
     * @param[out] w_d the `w` vector to assemble
     * @param[in] data_d the one-dimension support vector matrix
     * @param[in] data_last_d the last row of the support vector matrix
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_data_points the total number of support vectors
     * @param[in] num_features the number of features per support vector
     */
    device_kernel_w_linear(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const kernel_index_type num_features) :
        w_d_{ w_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, num_features_{ num_features } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<1> nd_idx) const {
        const kernel_index_type index = nd_idx.get_global_linear_id();
        real_type temp = 0;
        if (index < num_features_) {
            for (kernel_index_type dat = 0; dat < num_data_points_ - 1; ++dat) {
                temp += alpha_d_[dat] * data_d_[dat + (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
            }
            temp += alpha_d_[num_data_points_ - 1] * data_last_d_[index];
            w_d_[index] = temp;
        }
    }

  private:
    real_type *w_d_;
    const real_type *data_d_;
    const real_type *data_last_d_;
    const real_type *alpha_d_;
    const kernel_index_type num_data_points_;
    const kernel_index_type num_features_;
};

/**
 * @brief Predicts the labels for data points using the polynomial kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data points
 */
template <typename T>
class device_kernel_predict_poly {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel to predict the labels for data points using the polynomial kernel function.
     * @details Currently only single GPU execution is supported.
     * @param[in] out_d the calculated predictions
     * @param[in] data_d the one-dimension support vector matrix
     * @param[in] data_last_d the last row of the support vector matrix
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_data_points the total number of support vectors
     * @param[in] points the data points to predict
     * @param[in] num_predict_points the total number of data points to predict
     * @param[in] num_features the number of features per support vector and point to predict
     * @param[in] degree the degree parameter used in the polynomial kernel function
     * @param[in] gamma the gamma parameter used in the polynomial kernel function
     * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
     */
    device_kernel_predict_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const int degree, const real_type gamma, const real_type coef0) :
        out_d_{ out_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, points_{ points }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const kernel_index_type data_point_index = nd_idx.get_global_id(0);
        const kernel_index_type predict_point_index = nd_idx.get_global_id(1);

        real_type temp = 0;
        if (predict_point_index < num_predict_points_) {
            for (kernel_index_type feature_index = 0; feature_index < num_features_; ++feature_index) {
                if (data_point_index == num_data_points_) {
                    temp += data_last_d_[feature_index] * points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
                } else {
                    temp += data_d_[data_point_index + (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
                }
            }

            temp = alpha_d_[data_point_index] * ::sycl::pow(gamma_ * temp + coef0_, static_cast<real_type>(degree_));

            atomic_op<real_type>{ out_d_[predict_point_index] } += temp;
        }
    }

  private:
    real_type *out_d_;
    const real_type *data_d_;
    const real_type *data_last_d_;
    const real_type *alpha_d_;
    const kernel_index_type num_data_points_;
    const real_type *points_;
    const kernel_index_type num_predict_points_;
    const kernel_index_type num_features_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
};

/**
 * @brief Predicts the labels for data points using the radial basis functions kernel function.
 * @details Currently only single GPU execution is supported.
 * @tparam T the type of the data points
 */
template <typename T>
class device_kernel_predict_radial {
  public:
    /// The type of the data.
    using real_type = T;

    /**
     * @brief Construct a new device kernel to predict the labels for data points using the radial basis function kernel function.
     * @details Currently only single GPU execution is supported.
     * @param[in] out_d the calculated predictions
     * @param[in] data_d the one-dimension support vector matrix
     * @param[in] data_last_d the last row of the support vector matrix
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_data_points the total number of support vectors
     * @param[in] points the data points to predict
     * @param[in] num_predict_points the total number of data points to predict
     * @param[in] num_features the number of features per support vector and point to predict
     * @param[in] gamma the gamma parameter used in the rbf kernel function
     */
    device_kernel_predict_radial(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const kernel_index_type num_data_points, const real_type *points, const kernel_index_type num_predict_points, const kernel_index_type num_features, const real_type gamma) :
        out_d_{ out_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, points_{ points }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx the [`sycl::item`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#subsec:item.class)
     *                   identifying an instance of the functor executing at each point in a [`sycl::range`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#range-class)
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const kernel_index_type data_point_index = nd_idx.get_global_id(0);
        const kernel_index_type predict_point_index = nd_idx.get_global_id(1);

        real_type temp = 0;
        if (predict_point_index < num_predict_points_) {
            for (kernel_index_type feature_index = 0; feature_index < num_features_; ++feature_index) {
                if (data_point_index == num_data_points_) {
                    temp += (data_last_d_[feature_index] - points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_last_d_[feature_index] - points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
                } else {
                    temp += (data_d_[data_point_index + (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_d_[data_point_index + (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points_[predict_point_index + (num_predict_points_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
                }
            }

            temp = alpha_d_[data_point_index] * ::sycl::exp(-gamma_ * temp);

            atomic_op<real_type>{ out_d_[predict_point_index] } += temp;
        }
    }

  private:
    real_type *out_d_;
    const real_type *data_d_;
    const real_type *data_last_d_;
    const real_type *alpha_d_;
    const kernel_index_type num_data_points_;
    const real_type *points_;
    const kernel_index_type num_predict_points_;
    const kernel_index_type num_features_;
    const real_type gamma_;
};

}  // namespace plssvm::sycl
