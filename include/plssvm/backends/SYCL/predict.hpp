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
#include "plssvm/constants.hpp"                     // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"  // sycl::nd_item, sycl::range, sycl::pow, sycl::exp

namespace plssvm::sycl {

/// Unsigned integer type.
using size_type = std::size_t;

template <typename real_type>
class kernel_w {
  public:
    kernel_w(real_type *w_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const size_type num_data_points, const size_type num_features) :
        w_d_{ w_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, num_features_{ num_features } {}
    void operator()(::sycl::nd_item<1> nd_idx) const {
        const auto index = nd_idx.get_global_linear_id();
        real_type temp = 0;
        if (index < num_features_) {
            for (size_type dat = 0; dat < num_data_points_ - 1; ++dat) {
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
    const size_type num_data_points_;
    const size_type num_features_;
};

template <typename real_type>
class predict_points_poly {
  public:
    predict_points_poly(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const size_type num_data_points, const real_type *points, const size_type num_predict_points, const size_type num_features, const int degree, const real_type gamma, const real_type coef0) :
        out_d_{ out_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, points_{ points }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    void operator()(::sycl::nd_item<2> nd_idx) const {
        const size_type data_point_index = nd_idx.get_global_id(0);
        const size_type predict_point_index = nd_idx.get_global_id(1);

        real_type temp = 0;
        if (predict_point_index < num_predict_points_) {
            for (size_type feature_index = 0; feature_index < num_features_; ++feature_index) {
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
    const size_type num_data_points_;
    const real_type *points_;
    const size_type num_predict_points_;
    const size_type num_features_;
    const int degree_;
    const real_type gamma_;
    const real_type coef0_;
};

template <typename real_type>
class predict_points_rbf {
  public:
    predict_points_rbf(real_type *out_d, const real_type *data_d, const real_type *data_last_d, const real_type *alpha_d, const size_type num_data_points, const real_type *points, const size_type num_predict_points, const size_type num_features, const real_type gamma) :
        out_d_{ out_d }, data_d_{ data_d }, data_last_d_{ data_last_d }, alpha_d_{ alpha_d }, num_data_points_{ num_data_points }, points_{ points }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, gamma_{ gamma } {}

    void operator()(::sycl::nd_item<2> nd_idx) const {
        const size_type data_point_index = nd_idx.get_global_id(0);
        const size_type predict_point_index = nd_idx.get_global_id(1);

        real_type temp = 0;
        if (predict_point_index < num_predict_points_) {
            for (size_type feature_index = 0; feature_index < num_features_; ++feature_index) {
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
    const size_type num_data_points_;
    const real_type *points_;
    const size_type num_predict_points_;
    const size_type num_features_;
    const real_type gamma_;
};

}  // namespace plssvm::sycl
