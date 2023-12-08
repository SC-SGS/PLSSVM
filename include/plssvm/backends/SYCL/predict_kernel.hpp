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
#include "plssvm/constants.hpp"                     // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::item, sycl::pown, sycl::exp

namespace plssvm::sycl::detail {

/**
 * @brief Calculate the `q` vector used to speedup the prediction using the linear kernel function.
 */
class device_kernel_w_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[in,out] w_d the vector to speedup the linear prediction
     * @param[in] alpha_d the previously learned weights
     * @param[in] sv_d the support vectors
     * @param[in] num_classes the number of classes
     * @param[in] num_sv the number of support vectors
     */
    device_kernel_w_linear(::sycl::handler &cgh, real_type *w_d, const real_type *alpha_d, const real_type *sv_d, const unsigned long long num_classes, const unsigned long long num_sv) :
        data_cache_feature_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, data_cache_alpha_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, w_d_{ w_d }, alpha_d_{ alpha_d }, sv_d_{ sv_d }, num_classes_{ num_classes }, num_sv_{ num_sv } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long class_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long class_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long feature_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long feature_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long sv = 0; sv < num_sv_; sv += THREAD_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_feature_idx = feature_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_alpha_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = alpha_d_[global_class_idx * (num_sv_ + PADDING_SIZE) + sv + nd_idx.get_local_id(0)];   // AoS
                data_cache_feature_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = sv_d_[global_feature_idx * (num_sv_ + PADDING_SIZE) + sv + nd_idx.get_local_id(0)];  // SoA
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < THREAD_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_feature][internal_class] += data_cache_alpha_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_feature_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_feature];
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_feature = 0; internal_feature < INTERNAL_BLOCK_SIZE; ++internal_feature) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const unsigned long long global_class_idx = class_idx + internal_class;
                const unsigned long long global_feature_idx = feature_idx + internal_feature;

                w_d_[global_feature_idx * (num_classes_ + PADDING_SIZE) + global_class_idx] = temp[internal_feature][internal_class];
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_feature_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_alpha_;

    /// @cond Doxygen_suppress
    real_type *w_d_;
    const real_type *alpha_d_;
    const real_type *sv_d_;
    const unsigned long long num_classes_;
    const unsigned long long num_sv_;
    /// @endcond
};

/**
 * @brief Predict the @p predict_points_d using the linear kernel speeding up the calculation using the @p w_d vector.
 */
class device_kernel_predict_linear {
  public:
    /**
     * @brief Initialize the SYCL kernel function object.
     * @param[in] cgh the SYCL handler used to allocate the local memory
     * @param[out] out_d the predicted values
     * @param[in] w_d the vector to speedup the calculations
     * @param[in] rho_d the previously learned bias
     * @param[in] predict_points_d the data points to predict
     * @param[in] num_classes the number of classes
     * @param[in] num_predict_points the number of data points to predict
     * @param[in] num_features the number of features per data point
     */
    device_kernel_predict_linear(::sycl::handler &cgh, real_type *out_d, const real_type *w_d, const real_type *rho_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_predict_points, const unsigned long long num_features) :
        data_cache_pp_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, data_cache_w_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, out_d_{ out_d }, w_d_{ w_d }, rho_d_{ rho_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_predict_points_{ num_predict_points }, num_features_{ num_features } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long class_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long class_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long pp_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long pp_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_class_idx = class_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_w_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = w_d_[(dim + nd_idx.get_local_id(0)) * (num_classes_ + PADDING_SIZE) + global_class_idx];
                data_cache_w_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = w_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_classes_ + PADDING_SIZE) + global_class_idx];
                data_cache_pp_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0)) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
                data_cache_pp_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                        temp[internal_pd][internal_class] += data_cache_w_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_class] * data_cache_pp_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_pd];
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_class = 0; internal_class < INTERNAL_BLOCK_SIZE; ++internal_class) {
                const unsigned long long global_class_idx = class_idx + internal_class;
                const unsigned long long global_pp_idx = pp_idx + internal_pd;

                out_d_[global_pp_idx * (num_classes_ + PADDING_SIZE) + global_class_idx] = temp[internal_pd][internal_class] - rho_d_[global_class_idx];
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_pp_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_w_;

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
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_predict_polynomial(::sycl::handler &cgh, real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const int degree, const real_type gamma, const real_type coef0) :
        data_cache_pp_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, data_cache_sv_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, out_d_{ out_d }, alpha_d_{ alpha_d }, rho_d_{ rho_d }, sv_d_{ sv_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_sv_{ num_sv }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long sv_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long sv_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long pp_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long pp_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_sv_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = sv_d_[(dim + nd_idx.get_local_id(0)) * (num_sv_ + PADDING_SIZE) + global_sv_idx];
                data_cache_sv_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = sv_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_sv_ + PADDING_SIZE) + global_sv_idx];
                data_cache_pp_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0)) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
                data_cache_pp_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        temp[internal_pd][internal_sv] += data_cache_sv_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_sv] * data_cache_pp_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_pd];
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                const unsigned long long global_sv_idx = sv_idx + internal_sv;
                const unsigned long long global_pp_idx = pp_idx + internal_pd;

                const real_type temp_pp_sv = temp[internal_pd][internal_sv];
                for (unsigned long long class_idx = 0; class_idx < num_classes_; ++class_idx) {
                    // apply degree, gamma, and coef0, alpha and rho
                    real_type class_temp = alpha_d_[class_idx * (num_sv_ + PADDING_SIZE) + global_sv_idx] * ::sycl::pown(gamma_ * temp_pp_sv + coef0_, degree_);
                    if (global_sv_idx == 0) {
                        class_temp -= rho_d_[class_idx];
                    }

                    detail::atomic_op<real_type>{ out_d_[global_pp_idx * (num_classes_ + PADDING_SIZE) + class_idx] } += class_temp;
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_pp_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_sv_;

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
     * @param[in] cgh the SYCL handler used to allocate the local memory
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
    device_kernel_predict_rbf(::sycl::handler &cgh, real_type *out_d, const real_type *alpha_d, const real_type *rho_d, const real_type *sv_d, const real_type *predict_points_d, const unsigned long long num_classes, const unsigned long long num_sv, const unsigned long long num_predict_points, const unsigned long long num_features, const real_type gamma) :
        data_cache_pp_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, data_cache_sv_{ ::sycl::range<2>{ FEATURE_BLOCK_SIZE, INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE }, cgh }, out_d_{ out_d }, alpha_d_{ alpha_d }, rho_d_{ rho_d }, sv_d_{ sv_d }, predict_points_d_{ predict_points_d }, num_classes_{ num_classes }, num_sv_{ num_sv }, num_predict_points_{ num_predict_points }, num_features_{ num_features }, gamma_{ gamma } {}

    /**
     * @brief Function call operator overload performing the actual calculation.
     * @param[in] nd_idx indices representing the current point in the execution space
     */
    void operator()(::sycl::nd_item<2> nd_idx) const {
        const unsigned long long sv_idx = nd_idx.get_global_id(0) * INTERNAL_BLOCK_SIZE;
        const unsigned long long sv_cached_idx_linear = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);
        const unsigned long long pp_idx = nd_idx.get_global_id(1) * INTERNAL_BLOCK_SIZE;
        const unsigned long long pp_idx_linear = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE + nd_idx.get_local_id(1);

        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { { 0.0 } };

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const unsigned long long global_sv_idx = sv_cached_idx_linear + internal * THREAD_BLOCK_SIZE;
                const unsigned long long global_pp_idx = pp_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_sv_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = sv_d_[(dim + nd_idx.get_local_id(0)) * (num_sv_ + PADDING_SIZE) + global_sv_idx];
                data_cache_sv_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = sv_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_sv_ + PADDING_SIZE) + global_sv_idx];
                data_cache_pp_[nd_idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0)) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
                data_cache_pp_[nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + nd_idx.get_local_id(1)] = predict_points_d_[(dim + nd_idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_predict_points_ + PADDING_SIZE) + global_pp_idx];
            }
            nd_idx.barrier();

            // calculation
            for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
                    for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                        const real_type d = data_cache_sv_[block_dim][nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_sv] - data_cache_pp_[block_dim][nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_pd];
                        temp[internal_pd][internal_sv] += d * d;
                    }
                }
            }
            nd_idx.barrier();
        }

        for (unsigned internal_pd = 0; internal_pd < INTERNAL_BLOCK_SIZE; ++internal_pd) {
            for (unsigned internal_sv = 0; internal_sv < INTERNAL_BLOCK_SIZE; ++internal_sv) {
                const unsigned long long global_sv_idx = sv_idx + internal_sv;
                const unsigned long long global_pp_idx = pp_idx + internal_pd;

                const real_type temp_pp_sv = temp[internal_pd][internal_sv];
                for (unsigned long long class_idx = 0; class_idx < num_classes_; ++class_idx) {
                    // apply gamma, alpha and rho
                    real_type class_temp = alpha_d_[class_idx * (num_sv_ + PADDING_SIZE) + global_sv_idx] * ::sycl::exp(-gamma_ * temp_pp_sv);
                    if (global_sv_idx == 0) {
                        class_temp -= rho_d_[class_idx];
                    }

                    detail::atomic_op<real_type>{ out_d_[global_pp_idx * (num_classes_ + PADDING_SIZE) + class_idx] } += class_temp;
                }
            }
        }
    }

  private:
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_pp_;
    /// Local memory used for internal memory access optimizations.
    ::sycl::local_accessor<real_type, 2> data_cache_sv_;

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