/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/SYCL/q_kernel.hpp"

#include "sycl/sycl.hpp"  // sycl::item, sycl::pow, sycl::exp

namespace plssvm::sycl {

template <typename T>
device_kernel_q_linear<T>::device_kernel_q_linear(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int first_feature, const int last_feature) :
    q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, first_feature_{ first_feature }, last_feature_{ last_feature } {}

template <typename T>
void device_kernel_q_linear<T>::operator()(::sycl::item<1> item) const {
    const auto index = item.get_linear_id();
    real_type temp{ 0.0 };
    for (int i = first_feature_; i < last_feature_; ++i) {
        temp += data_d_[i * num_rows_ + index] * data_last_[i];
    }
    q_[index] = temp;
}

template class device_kernel_q_linear<float>;
template class device_kernel_q_linear<double>;

template <typename T>
device_kernel_q_poly<T>::device_kernel_q_poly(T *q, const T *data_d, const T *data_last, const int num_rows, const int num_cols, const T degree, const T gamma, const T coef0) :
    q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

template <typename T>
void device_kernel_q_poly<T>::operator()(::sycl::item<1> item) const {
    const auto index = item.get_linear_id();
    real_type temp{ 0.0 };
    for (int i = 0; i < num_cols_; ++i) {
        temp += data_d_[i * num_rows_ + index] * data_last_[i];
    }
    q_[index] = ::sycl::pow(gamma_ * temp + coef0_, degree_);
}

template class device_kernel_q_poly<float>;
template class device_kernel_q_poly<double>;

template <typename T>
device_kernel_q_radial<T>::device_kernel_q_radial(real_type *q, const real_type *data_d, const real_type *data_last, const int num_rows, const int num_cols, const real_type gamma) :
    q_{ q }, data_d_{ data_d }, data_last_{ data_last }, num_rows_{ num_rows }, num_cols_{ num_cols }, gamma_{ gamma } {}

template <typename T>
void device_kernel_q_radial<T>::operator()(::sycl::item<1> item) const {
    const auto index = item.get_linear_id();
    real_type temp{ 0.0 };
    for (int i = 0; i < num_cols_; ++i) {
        temp += (data_d_[i * num_rows_ + index] - data_last_[i]) * (data_d_[i * num_rows_ + index] - data_last_[i]);
    }
    q_[index] = ::sycl::exp(-gamma_ * temp);
}

template class device_kernel_q_radial<float>;
template class device_kernel_q_radial<double>;

}  // namespace plssvm::sycl