/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assembling the kernel matrix using the SYCL backend with hierarchical kernels.
 */

#ifndef PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIERARCHICAL_HPP_
#define PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIERARCHICAL_HPP_
#pragma once

#include "plssvm/constants.hpp"  // plssvm::{real_type, THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE, FEATURE_BLOCK_SIZE, PADDING_SIZE}

#include "sycl/sycl.hpp"  // sycl::h_item, sycl::pown, sycl::exp

namespace plssvm::sycl::detail::hierarchical {

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
     * @param[in] group the hierarchical group representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<unsigned long long, 2> private_i{ group };
        ::sycl::private_memory<unsigned long long, 2> private_i_cached_idx_linear{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j_linear{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_temp{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = (group[0] * group.get_local_range(0) + idx.get_local_id(0)) * INTERNAL_BLOCK_SIZE;
            private_i_cached_idx_linear(idx) = group[0] * group.get_local_range(0) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_j(idx) = (group[1] * group.get_local_range(1) + idx.get_local_id(1)) * INTERNAL_BLOCK_SIZE;
            private_j_linear(idx) = group[1] * group.get_local_range(1) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_cond(idx) = group[0] >= group[1];

            // matrix
            for (unsigned i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                for (unsigned j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_temp(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = private_i_cached_idx_linear(idx) + internal * THREAD_BLOCK_SIZE;
                        const unsigned long long global_j = private_j_linear(idx) + internal * THREAD_BLOCK_SIZE;

                        data_cache_i[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_i[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_j[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                        data_cache_j[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    }
                }
            });

            // implicit group barrier

            // perform calculations
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                private_temp(idx)[internal_i][internal_j] += data_cache_i[block_dim][idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                            }
                        }
                    }
                }
            });

            // implicit barrier
        }

        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const unsigned long long global_i = private_i(idx) + internal_i;
                        const unsigned long long global_j = private_j(idx) + internal_j;

                        if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                            real_type temp_ij = private_temp(idx)[internal_i][internal_j];
                            temp_ij = temp_ij + QA_cost_ - q_[global_i] - q_[global_j];
                            if (global_i == global_j) {
                                temp_ij += cost_;
                            }

#if defined(PLSSVM_USE_GEMM)
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i] = temp_ij;
                            ret_[global_i * (num_rows_ + PADDING_SIZE) + global_j] = temp_ij;
#else
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                        }
                    }
                }
            }
        });
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
     * @param[in] group the hierarchical group representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<unsigned long long, 2> private_i{ group };
        ::sycl::private_memory<unsigned long long, 2> private_i_cached_idx_linear{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j_linear{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_temp{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = (group[0] * group.get_local_range(0) + idx.get_local_id(0)) * INTERNAL_BLOCK_SIZE;
            private_i_cached_idx_linear(idx) = group[0] * group.get_local_range(0) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_j(idx) = (group[1] * group.get_local_range(1) + idx.get_local_id(1)) * INTERNAL_BLOCK_SIZE;
            private_j_linear(idx) = group[1] * group.get_local_range(1) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_cond(idx) = group[0] >= group[1];

            // matrix
            for (unsigned i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                for (unsigned j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_temp(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = private_i_cached_idx_linear(idx) + internal * THREAD_BLOCK_SIZE;
                        const unsigned long long global_j = private_j_linear(idx) + internal * THREAD_BLOCK_SIZE;

                        data_cache_i[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_i[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_j[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                        data_cache_j[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    }
                }
            });

            // implicit group barrier

            // perform calculations
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                private_temp(idx)[internal_i][internal_j] += data_cache_i[block_dim][idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                            }
                        }
                    }
                }
            });

            // implicit barrier
        }

        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const unsigned long long global_i = private_i(idx) + internal_i;
                        const unsigned long long global_j = private_j(idx) + internal_j;

                        if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                            real_type temp_ij = private_temp(idx)[internal_i][internal_j];
                            temp_ij = ::sycl::pown(gamma_ * temp_ij + coef0_, degree_) + QA_cost_ - q_[global_i] - q_[global_j];
                            if (global_i == global_j) {
                                temp_ij += cost_;
                            }

#if defined(PLSSVM_USE_GEMM)
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i] = temp_ij;
                            ret_[global_i * (num_rows_ + PADDING_SIZE) + global_j] = temp_ij;
#else
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                        }
                    }
                }
            }
        });
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
     * @param[in] group the hierarchical group representing the current point in the execution space
     */
    void operator()(::sycl::group<2> group) const {
        // allocate shared memory
        real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
        real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

        // allocate memory for work-item local variables
        // -> accessible across different 'parallel_for_work_item' invocations
        ::sycl::private_memory<unsigned long long, 2> private_i{ group };
        ::sycl::private_memory<unsigned long long, 2> private_i_cached_idx_linear{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j{ group };
        ::sycl::private_memory<unsigned long long, 2> private_j_linear{ group };
        ::sycl::private_memory<real_type[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE], 2> private_temp{ group };
        ::sycl::private_memory<bool, 2> private_cond{ group };

        // initialize private and local variables
        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            // indices and diagonal condition
            private_i(idx) = (group[0] * group.get_local_range(0) + idx.get_local_id(0)) * INTERNAL_BLOCK_SIZE;
            private_i_cached_idx_linear(idx) = group[0] * group.get_local_range(0) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_j(idx) = (group[1] * group.get_local_range(1) + idx.get_local_id(1)) * INTERNAL_BLOCK_SIZE;
            private_j_linear(idx) = group[1] * group.get_local_range(1) * INTERNAL_BLOCK_SIZE + idx.get_local_id(1);
            private_cond(idx) = group[0] >= group[1];

            // matrix
            for (unsigned i = 0; i < INTERNAL_BLOCK_SIZE; ++i) {
                for (unsigned j = 0; j < INTERNAL_BLOCK_SIZE; ++j) {
                    private_temp(idx)[i][j] = real_type{ 0.0 };
                }
            }
        });

        // implicit group barrier

        for (unsigned long long dim = 0; dim < num_features_; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                        const unsigned long long global_i = private_i_cached_idx_linear(idx) + internal * THREAD_BLOCK_SIZE;
                        const unsigned long long global_j = private_j_linear(idx) + internal * THREAD_BLOCK_SIZE;

                        data_cache_i[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_i[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_i];
                        data_cache_j[idx.get_local_id(0)][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0)) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                        data_cache_j[idx.get_local_id(0) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + idx.get_local_id(1)] = data_d_[(dim + idx.get_local_id(0) + THREAD_BLOCK_SIZE) * (num_rows_ + 1 + PADDING_SIZE) + global_j];
                    }
                }
            });

            // implicit group barrier

            // perform calculations
            group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
                if (private_cond(idx)) {
                    for (unsigned block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                        for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                            for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                                const real_type d = data_cache_i[block_dim][idx.get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] - data_cache_j[block_dim][idx.get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                                private_temp(idx)[internal_i][internal_j] += d * d;
                            }
                        }
                    }
                }
            });

            // implicit barrier
        }

        group.parallel_for_work_item([&](::sycl::h_item<2> idx) {
            if (private_cond(idx)) {
                for (unsigned internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (unsigned internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const unsigned long long global_i = private_i(idx) + internal_i;
                        const unsigned long long global_j = private_j(idx) + internal_j;

                        if (global_i < num_rows_ && global_j < num_rows_ && global_i >= global_j) {
                            real_type temp_ij = private_temp(idx)[internal_i][internal_j];
                            temp_ij = ::sycl::exp(-gamma_ * temp_ij) + QA_cost_ - q_[global_i] - q_[global_j];
                            if (global_i == global_j) {
                                temp_ij += cost_;
                            }

#if defined(PLSSVM_USE_GEMM)
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i] = temp_ij;
                            ret_[global_i * (num_rows_ + PADDING_SIZE) + global_j] = temp_ij;
#else
                            ret_[global_j * (num_rows_ + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                        }
                    }
                }
            }
        });
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

}  // namespace plssvm::sycl::detail::hierarchical

#endif  // PLSSVM_BACKENDS_SYCL_CG_EXPLICIT_KERNEL_MATRIX_ASSEMBLY_HIERARCHICAL_HPP_
