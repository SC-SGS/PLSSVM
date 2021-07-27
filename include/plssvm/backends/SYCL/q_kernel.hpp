/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines SYCL functions for generating the `q` vector.
 */

#pragma once

#include "sycl/sycl.hpp"

namespace plssvm::sycl {

template <typename real_type>
class device_kernel_q_linear {
  public:
    device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int first_feature, const int last_feature) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, first_feature_{ first_feature }, last_feature_{ last_feature } {}

    void operator()(::sycl::item<1> item) const {
        const auto index = item.get_linear_id();
        real_type temp{ 0.0 };
        for (int i = first_feature_; i < last_feature_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = temp;
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int first_feature_;
    const int last_feature_;
};

template <typename real_type>
class device_kernel_q_poly {
  public:
    device_kernel_q_poly(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int num_cols, const real_type degree, const real_type gamma, const real_type coef0) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    void operator()(::sycl::item<1> item) const {
        const auto index = item.get_linear_id();
        real_type temp{ 0.0 };
        for (int i = 0; i < num_cols_; ++i) {
            temp += data_d_[i * num_rows_ + index] * data_last_[i];
        }
        q_[index] = ::sycl::pow(gamma_ * temp + coef0_, degree_);
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const real_type degree_;
    const real_type gamma_;
    const real_type coef0_;
};

template <typename real_type>
class device_kernel_q_radial {
  public:
    device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int num_cols, const real_type gamma) :
        q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, gamma_{ gamma } {}

    void operator()(::sycl::item<1> item) const {
        const auto index = item.get_linear_id();
        real_type temp{ 0.0 };
        for (int i = 0; i < num_cols_; ++i) {
            temp += (data_d_[i * num_rows_ + index] - data_last_[i]) * (data_d_[i * num_rows_ + index] - data_last_[i]);
        }
        q_[index] = ::sycl::exp(-gamma_ * temp);
    }

  private:
    real_type *q_;
    const real_type *data_d_;
    const real_type *data_last_;
    const int num_rows_;
    const int num_cols_;
    const real_type gamma_;
};

}  // namespace plssvm::sycl