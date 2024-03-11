/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends using a GPU. Used for code duplication reduction.
 */

#ifndef PLSSVM_BACKENDS_GPU_CSVM_HPP_
#define PLSSVM_BACKENDS_GPU_CSVM_HPP_
#pragma once

#include "plssvm/constants.hpp"                 // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/csvm.hpp"                      // plssvm::csvm
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"  // plssvm::detail::{data_distribution, triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/move_only_any.hpp"      // plssvm::detail::{move_only_any, move_only_any_cast}
#include "plssvm/detail/operators.hpp"          // operator namespace
#include "plssvm/kernel_function_types.hpp"     // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                    // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/shape.hpp"                     // plssvm::shape
#include "plssvm/solver_types.hpp"              // plssvm::solver_type

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min, std::all_of
#include <cstddef>    // std::size_t
#include <memory>     // std::unique_ptr, std::make_unique
#include <tuple>      // std::tuple
#include <utility>    // std::forward, std::move
#include <vector>     // std::vector

namespace plssvm::detail {

/**
 * @brief A C-SVM implementation for all GPU backends to reduce code duplication.
 * @details Implements all virtual functions defined in plssvm::csvm. The GPU backends only have to implement the actual kernel (launches).
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 * @tparam pinned_memory_t the type of the pinned memory wrapper (dependent on the used backend)
 */
template <template <typename> typename device_ptr_t, typename queue_t, template <typename> typename pinned_memory_t>
class gpu_csvm : public ::plssvm::csvm {
  public:
    /// The type of the device pointer (dependent on the used backend).
    using device_ptr_type = device_ptr_t<real_type>;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;
    /// The type of the pinned memory (dependent on the used backend).
    using pinned_memory_type = pinned_memory_t<real_type>;

    /**
     * @copydoc plssvm::csvm::csvm()
     */
    explicit gpu_csvm(parameter params = {}) :
        ::plssvm::csvm{ params } { }

    /**
     * @brief Construct a C-SVM forwarding all parameters @p args to the plssvm::parameter constructor.
     * @tparam Args the type of the (named-)parameters
     * @param[in] args the parameters used to construct a plssvm::parameter
     */
    template <typename... Args>
    explicit gpu_csvm(Args &&...args) :
        ::plssvm::csvm{ std::forward<Args>(args)... } { }

    /**
     * @copydoc plssvm::csvm::csvm(const plssvm::csvm &)
     */
    gpu_csvm(const gpu_csvm &) = delete;
    /**
     * @copydoc plssvm::csvm::csvm(plssvm::csvm &&) noexcept
     */
    gpu_csvm(gpu_csvm &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::operator=(const plssvm::csvm &)
     */
    gpu_csvm &operator=(const gpu_csvm &) = delete;
    /**
     * @copydoc plssvm::csvm::operator=(plssvm::csvm &&) noexcept
     */
    gpu_csvm &operator=(gpu_csvm &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::~csvm()
     */
    ~gpu_csvm() override = default;

    /**
     * plssvm::csvm::num_available_devices
     */
    [[nodiscard]] std::size_t num_available_devices() const noexcept override {
        return devices_.size();
    }

  protected:
    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    [[nodiscard]] std::vector<::plssvm::detail::move_only_any> assemble_kernel_matrix(solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::blas_level_3
     */
    void blas_level_3(solver_type solver, real_type alpha, const std::vector<::plssvm::detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const final;

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    // Note: there are two versions of each function (one for float and one for double) since virtual template functions are not allowed in C++!

    /**
     * @brief Return the maximum allowed work group size for the specified device.
     * @return the maximum allowed work group size (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::size_t get_max_work_group_size(std::size_t device_id) const = 0;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @brief Explicitly assemble the kernel matrix on the device with @p device_id.
     * @param[in] device_id the device to run the kernel on
     * @param[in] params the parameters (e.g., kernel function) used to assemble the kernel matrix
     * @param[in] data_d the data set to create the kernel matrix from
     * @param[in] q_red_d the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @return the explicit kernel matrix stored on the device (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_assemble_kernel_matrix_explicit(std::size_t device_id, const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const = 0;
    /**
     * @brief Perform an explicit BLAS level 3 operation: `C = alpha * A * B + beta * C` where @p A, @p B, and @p C are matrices, and @p alpha and @p beta are scalars.
     * @param[in] device_id the device to run the kernel on
     * @param[in] alpha the scalar alpha value
     * @param[in] A_d the matrix @p A
     * @param[in] B_d the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C_d the matrix @p C, also used as result matrix
     */
    virtual void run_blas_level_3_kernel_explicit(std::size_t device_id, real_type alpha, const device_ptr_type &A_d, const device_ptr_type &B_d, real_type beta, device_ptr_type &C_d) const = 0;
    /**
     * @brief Perform an implicit BLAS level 3 operation: `C = alpha * A * B + beta * C` where @p A is an implicitly constructed kernel matrix, @p B and @p C are matrices and @p alpha is a scalar.
     * @param[in] device_id the device to run the kernel on
     * @param[in] alpha the scalar alpha value
     * @param[in] A_d the data points to implicitly create the kernel matrix from
     * @param[in] params the parameters (e.g., kernel function) used to assemble the kernel matrix
     * @param[in] q_red_d the vector used in the dimensional reduction of the kernel matrix
     * @param[in] QA_cost the scalar used in the dimensional reduction of the kernel matrix
     * @param[in] B_d the matrix @p B
     * @param[in,out] C_d the matrix @p C, also used as result matrix
     */
    virtual void run_assemble_kernel_matrix_implicit_blas_level_3(std::size_t device_id, real_type alpha, const device_ptr_type &A_d, const parameter &params, const device_ptr_type &q_red_d, real_type QA_cost, const device_ptr_type &B_d, device_ptr_type &C_d) const = 0;
    /**
     * @brief Perform a simple inplace matrix addition adding the contents of @p rhs_d to @p lhs_d.
     * @param[in] device_id the device to run the kernel on
     * @param[in,out] lhs_d the matrix to add the values of @p rhs_d to
     * @param[in] rhs_d the matrix to add to @p lhs_d
     */
    virtual void run_inplace_matrix_addition(std::size_t device_id, device_ptr_type &lhs_d, const device_ptr_type &rhs_d) const = 0;
    /**
     * @brief Perform a simple inplace matrix scale: @p lhs_d *= @p scale.
     * @param[in] device_id the device to run the kernel on
     * @param[in,out] lhs_d the matrix to @p scale
     * @param[in] scale the scaling value
     */
    virtual void run_inplace_matrix_scale(std::size_t device_id, device_ptr_type &lhs_d, real_type scale) const = 0;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @brief Calculate the `w` vector used to speedup the linear kernel prediction.
     * @param[in] device_id the device to run the kernel on
     * @param[in] alpha_d the support vector weights
     * @param[in] sv_d the support vectors
     * @return the `w` vector (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_w_kernel(std::size_t device_id, const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const = 0;
    /**
     * @brief Predict the values of the new @p predict_points_d using the previously learned weights @p alpha_d, biases @p rho_d, and support vectors @p sv_d.
     * @param[in] device_id the device to run the kernel on
     * @param[in] params the parameter used to predict the values (e.g., the used kernel function)
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously calculated biases
     * @param[in] sv_or_w_d the previously learned support vectors or the `w` used to speedup the linear kernel prediction (only used in the linear kernel case)
     * @param[in] predict_points_d the new data points to predict
     * @return the predicted values (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_predict_kernel(std::size_t device_id, const parameter &params, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_or_w_d, const device_ptr_type &predict_points_d) const = 0;

    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
};

//***************************************************//
//                        fit                        //
//***************************************************//
template <template <typename> typename device_ptr_t, typename queue_t, template <typename> typename pinned_memory_t>
std::vector<::plssvm::detail::move_only_any> gpu_csvm<device_ptr_t, queue_t, pinned_memory_t>::assemble_kernel_matrix(const solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, const real_type QA_cost) const {
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices must not be empty!");
    PLSSVM_ASSERT(A.is_padded(), "The matrix to setup on the devices must be padded!");
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector must not be empty!");
    PLSSVM_ASSERT(q_red.size() == A.num_rows() - 1, "The q_red size ({}) mismatches the number of data points after dimensional reduction ({})!", q_red.size(), A.num_rows() - 1);

    // update the data distribution: only the upper triangular kernel matrix is used
    // note: account for the dimensional reduction
    data_distribution_ = std::make_unique<detail::triangular_data_distribution>(A.num_rows() - 1, this->num_available_devices());

    // the final kernel matrix; multiple parts in case of multi-device execution
    std::vector<::plssvm::detail::move_only_any> kernel_matrices_parts(this->num_available_devices());
    // the input data and dimensional reduction vector; completely stored on each device
    std::vector<device_ptr_type> data_d(this->num_available_devices());
    std::vector<device_ptr_type> q_red_d(this->num_available_devices());

    // split memory allocation and memory copy! (necessary to remove locks on some systems and setups)
#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        const queue_type &device = devices_[device_id];

        // allocate memory on the device
        data_d[device_id] = device_ptr_type{ A.shape(), A.padding(), device };
        q_red_d[device_id] = device_ptr_type{ q_red.size() + PADDING_SIZE, device };
    }

    // pin the data matrix
    const pinned_memory_type pm{ A };

#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // copy data to the device
        data_d[device_id].copy_to_device(A);
        q_red_d[device_id].copy_to_device(q_red, 0, q_red.size());
        q_red_d[device_id].memset(0, q_red.size());

        if (solver == solver_type::cg_explicit) {
            // explicitly assemble the (potential partial) kernel matrix
            device_ptr_type kernel_matrix = this->run_assemble_kernel_matrix_explicit(device_id, params, data_d[device_id], q_red_d[device_id], QA_cost);
            kernel_matrices_parts[device_id] = ::plssvm::detail::move_only_any{ std::move(kernel_matrix) };
        } else if (solver == solver_type::cg_implicit) {
            // simply return the data since in cg_implicit we don't assemble the kernel matrix here!
            kernel_matrices_parts[device_id] = ::plssvm::detail::move_only_any{ std::make_tuple(std::move(data_d[device_id]), params, std::move(q_red_d[device_id]), QA_cost) };
        } else {
            throw exception{ fmt::format("Assembling the kernel matrix using the {} CG solver_type is currently not implemented!", solver) };
        }
    }

    return kernel_matrices_parts;
}

template <template <typename> typename device_ptr_t, typename queue_t, template <typename> typename pinned_memory_t>
void gpu_csvm<device_ptr_t, queue_t, pinned_memory_t>::blas_level_3(const solver_type solver, const real_type alpha, const std::vector<::plssvm::detail::move_only_any> &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");
    PLSSVM_ASSERT(A.size() == this->num_available_devices(), "Not enough kernel matrix parts ({}) for the available number of devices ({})!", A.size(), this->num_available_devices());
    PLSSVM_ASSERT(!B.empty(), "The B matrix must not be empty!");
    PLSSVM_ASSERT(B.is_padded(), "The B matrix must be padded!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix must not be empty!");
    PLSSVM_ASSERT(C.is_padded(), "The C matrix must be padded!");
    PLSSVM_ASSERT(B.shape() == C.shape(), "The B ({}) and C ({}) matrices must have the same shape!", B.shape(), C.shape());
    PLSSVM_ASSERT(B.padding() == C.padding(), "The B ({}) and C ({}) matrices must have the same padding!", B.padding(), C.padding());

    // the C and B matrices; completely stored on each device
    std::vector<device_ptr_type> B_d(this->num_available_devices());
    std::vector<device_ptr_type> C_d(this->num_available_devices());

    // the partial C result from a specific device later stored on device 0 to perform the C reduction (inplace matrix addition)
    device_ptr_type partial_C_d{};
    if (this->num_available_devices() > 1) {
        partial_C_d = device_ptr_type{ C.shape(), C.padding(), devices_[0] };
    }

    // split memory allocation and memory copy!
#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        const queue_type &device = devices_[device_id];

        // allocate memory on the device
        B_d[device_id] = device_ptr_type{ B.shape(), B.padding(), device };
        C_d[device_id] = device_ptr_type{ C.shape(), C.padding(), device };
    }

#pragma omp parallel for ordered
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }

        // copy data to the device
        B_d[device_id].copy_to_device(B);
        if (device_id == 0) {
            // device 0 always touches all values in C -> it is sufficient that only device 0 gets the actual C matrix
            C_d[device_id].copy_to_device(C);
            // we do not perform the beta scale in C in the cg_implicit device kernel
            // -> calculate it using a separate kernel (always on device 0!)
            if (solver == solver_type::cg_implicit) {
                this->run_inplace_matrix_scale(0, C_d[device_id], beta);
            }
        } else {
            // all other devices get a matrix filled with zeros only
            C_d[device_id].memset(0);
        }

        if (solver == solver_type::cg_explicit) {
            const auto &A_d = detail::move_only_any_cast<const device_ptr_type &>(A[device_id]);
            PLSSVM_ASSERT(!A_d.empty(), "The A matrix must not be empty!");

            this->run_blas_level_3_kernel_explicit(device_id, alpha, A_d, B_d[device_id], beta, C_d[device_id]);
        } else if (solver == solver_type::cg_implicit) {
            const auto &[A_d, params, q_red_d, QA_cost] = detail::move_only_any_cast<const std::tuple<device_ptr_type, parameter, device_ptr_type, real_type> &>(A[device_id]);
            PLSSVM_ASSERT(!A_d.empty(), "The A matrix must not be empty!");
            PLSSVM_ASSERT(!q_red_d.empty(), "The q_red vector must not be empty!");

            this->run_assemble_kernel_matrix_implicit_blas_level_3(device_id, alpha, A_d, params, q_red_d, QA_cost, B_d[device_id], C_d[device_id]);
        } else {
            throw exception{ fmt::format("The BLAS calculation using the {} CG solver_type is currently not implemented!", solver) };
        }

        // reduce the partial C matrices to the final C matrix on device 0
#pragma omp ordered
        if (device_id != 0) {
            C_d[device_id].copy_to_other_device(partial_C_d);
            // always reduce on device 0!!!
            this->run_inplace_matrix_addition(0, C_d[0], partial_C_d);
        }
    }

    // device 0 contains the final, reduced results
    C_d[0].copy_to_host(C);
    C.restore_padding();
}

//***************************************************//
//                   predict, score                  //
//***************************************************//
template <template <typename> typename device_ptr_t, typename queue_t, template <typename> typename pinned_memory_t>
aos_matrix<real_type> gpu_csvm<device_ptr_t, queue_t, pinned_memory_t>::predict_values(const parameter &params,
                                                                                       const soa_matrix<real_type> &support_vectors,
                                                                                       const aos_matrix<real_type> &alpha,
                                                                                       const std::vector<real_type> &rho,
                                                                                       soa_matrix<real_type> &w,
                                                                                       const soa_matrix<real_type> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(support_vectors.is_padded(), "The support vectors must be padded!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha vectors (weights) must not be empty!");
    PLSSVM_ASSERT(alpha.is_padded(), "The alpha vectors (weights) must be padded!");
    PLSSVM_ASSERT(support_vectors.num_rows() == alpha.num_cols(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.num_rows(), alpha.num_cols());
    PLSSVM_ASSERT(rho.size() == alpha.num_rows(), "The number of rho values ({}) and the number of weight vectors ({}) must be the same!", rho.size(), alpha.num_rows());
    PLSSVM_ASSERT(w.empty() || w.is_padded(), "Either w must be empty or must be padded!");
    PLSSVM_ASSERT(w.empty() || support_vectors.num_cols() == w.num_cols(), "Either w must be empty or contain exactly the same number of values ({}) as features are present ({})!", w.num_cols(), support_vectors.num_cols());
    PLSSVM_ASSERT(w.empty() || alpha.num_rows() == w.num_rows(), "Either w must be empty or contain exactly the same number of vectors ({}) as the alpha vector ({})!", w.num_rows(), alpha.num_rows());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(predict_points.is_padded(), "The data points to predict must be padded!");
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.num_cols(), predict_points.num_cols());

    // define necessary sizes
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    // the result matrix
    aos_matrix<real_type> out_ret{ shape{ num_predict_points, num_classes }, real_type{ 0.0 }, shape{ PADDING_SIZE, PADDING_SIZE } };

    // the support vectors or w vector and weights; fully stored on each device
    std::vector<device_ptr_type> sv_or_w_d(this->num_available_devices());
    std::vector<device_ptr_type> alpha_d(this->num_available_devices());

    // split memory allocation and memory copy!
#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        const queue_type &device = devices_[device_id];

        // allocate memory on the device
        alpha_d[device_id] = device_ptr_type{ alpha.shape(), alpha.padding(), device };
    }
#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // copy data to the device
        alpha_d[device_id].copy_to_device(alpha);
    }

    // special case for the linear kernel function: calculate or reuse the w vector to speed up the prediction
    if (params.kernel_type == kernel_function_type::linear) {
        if (w.empty()) {
            // the partial w result from a specific device later stored on device 0 to perform the w reduction (inplace matrix addition)
            std::vector<device_ptr_type> w_d(this->num_available_devices());
            device_ptr_type partial_w_d{};  // always on device 0!
            if (this->num_available_devices() > 1) {
                partial_w_d = device_ptr_type{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, devices_[0] };
            }

            // update the data distribution to account for the support vectors
            data_distribution_ = std::make_unique<detail::rectangular_data_distribution>(num_support_vectors, this->num_available_devices());

            std::vector<device_ptr_type> sv_d(this->num_available_devices());
            // split memory allocation and memory copy!
#pragma omp parallel for
            for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                // check whether the current device is responsible for at least one data point!
                if (data_distribution_->place_specific_num_rows(device_id) == 0) {
                    continue;
                }
                const queue_type &device = devices_[device_id];

                // allocate memory on the device
                sv_d[device_id] = device_ptr_type{ shape{ data_distribution_->place_specific_num_rows(device_id), num_features }, support_vectors.padding(), device };
            }

#pragma omp parallel for ordered
            for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
                // check whether the current device is responsible for at least one data point!
                if (data_distribution_->place_specific_num_rows(device_id) == 0) {
                    continue;
                }

                // copy data to the device
                sv_d[device_id].copy_to_device_strided(support_vectors, data_distribution_->place_row_offset(device_id), data_distribution_->place_specific_num_rows(device_id));

                // calculate the partial w vector
                w_d[device_id] = this->run_w_kernel(device_id, alpha_d[device_id], sv_d[device_id]);

                // reduce the partial w vectors on device 0
#pragma omp ordered
                if (device_id != 0) {
                    w_d[device_id].copy_to_other_device(partial_w_d);
                    // always reduce on device 0!!!
                    this->run_inplace_matrix_addition(0, w_d[0], partial_w_d);
                }
            }

            // w_d[0] contains the final w vector
            w = soa_matrix<real_type>{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE } };
            w_d[0].copy_to_host(w);
            w.restore_padding();
        }

        // upload the w vector to all devices
        // split memory allocation and memory copy!
#pragma omp parallel for
        for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
            const queue_type &device = devices_[device_id];

            // allocate memory on the device
            sv_or_w_d[device_id] = device_ptr_type{ shape{ num_classes, num_features }, shape{ PADDING_SIZE, PADDING_SIZE }, device };
        }
#pragma omp parallel for
        for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
            // copy data to the device
            sv_or_w_d[device_id].copy_to_device(w);
        }
    } else {
        // use the support vectors for all other kernel functions
        // split memory allocation and memory copy!
#pragma omp parallel for
        for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
            const queue_type &device = devices_[device_id];

            // allocate memory on the device
            sv_or_w_d[device_id] = device_ptr_type{ support_vectors.shape(), support_vectors.padding(), device };
        }
#pragma omp parallel for
        for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
            // copy data to the device
            sv_or_w_d[device_id].copy_to_device(support_vectors);
        }
    }

    // update the data distribution to account for the predict points
    data_distribution_ = std::make_unique<detail::rectangular_data_distribution>(num_predict_points, this->num_available_devices());

    // the predict points; partial stored on each device
    std::vector<device_ptr_type> predict_points_d(this->num_available_devices());
    // the biases; completely stored on each device
    std::vector<device_ptr_type> rho_d(this->num_available_devices());

    // split memory allocation and memory copy!
#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        const queue_type &device = devices_[device_id];
        const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);

        // allocate memory on the device
        predict_points_d[device_id] = device_ptr_type{ shape{ device_specific_num_rows, num_features }, predict_points.padding(), device };
        rho_d[device_id] = device_ptr_type{ num_classes + PADDING_SIZE, device };
    }

#pragma omp parallel for
    for (std::size_t device_id = 0; device_id < this->num_available_devices(); ++device_id) {
        // check whether the current device is responsible for at least one data point!
        if (data_distribution_->place_specific_num_rows(device_id) == 0) {
            continue;
        }
        const std::size_t device_specific_num_rows = data_distribution_->place_specific_num_rows(device_id);
        const std::size_t row_offset = data_distribution_->place_row_offset(device_id);

        // copy data to the device
        predict_points_d[device_id].copy_to_device_strided(predict_points, row_offset, device_specific_num_rows);
        rho_d[device_id].copy_to_device(rho, 0, rho.size());
        rho_d[device_id].memset(0, rho.size());

        // predict
        const device_ptr_type out_d = this->run_predict_kernel(device_id, params, alpha_d[device_id], rho_d[device_id], sv_or_w_d[device_id], predict_points_d[device_id]);

        // copy results back to host, combining them into one result matrix
#pragma omp critical
        out_d.copy_to_host(out_ret.data() + row_offset * (num_classes + PADDING_SIZE), 0, device_specific_num_rows * (num_classes + PADDING_SIZE));
    }

    out_ret.restore_padding();
    return out_ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_CSVM_HPP_
