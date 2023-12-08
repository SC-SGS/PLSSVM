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

#include "plssvm/constants.hpp"     // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/csvm.hpp"          // plssvm::csvm
#include "plssvm/matrix.hpp"        // plssvm::aos_matrix
#include "plssvm/parameter.hpp"     // plssvm::parameter
#include "plssvm/solver_types.hpp"  // plssvm::solver_type

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min, std::all_of
#include <cstddef>    // std::size_t
#include <utility>    // std::forward, std::move
#include <vector>     // std::vector

namespace plssvm::detail {

/**
 * @brief A C-SVM implementation for all GPU backends to reduce code duplication.
 * @details Implements all virtual functions defined in plssvm::csvm. The GPU backends only have to implement the actual kernel (launches).
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 */
template <template <typename> typename device_ptr_t, typename queue_t>
class gpu_csvm : public ::plssvm::csvm {
  public:
    /// The type of the device pointer (dependent on the used backend).
    using device_ptr_type = device_ptr_t<real_type>;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;

    /**
     * @copydoc plssvm::csvm::csvm()
     */
    explicit gpu_csvm(parameter params = {}) :
        ::plssvm::csvm{ params } {}
    /**
     * @brief Construct a C-SVM forwarding all parameters @p args to the plssvm::parameter constructor.
     * @tparam Args the type of the (named-)parameters
     * @param[in] args the parameters used to construct a plssvm::parameter
     */
    template <typename... Args>
    explicit gpu_csvm(Args &&...args) :
        ::plssvm::csvm{ std::forward<Args>(args)... } {}

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
     * @brief Return the number of available devices for the current backend.
     * @return the number of available devices (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_available_devices() const noexcept {
        return devices_.size();
    }

  protected:
    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::setup_data_on_devices
     */
    [[nodiscard]] ::plssvm::detail::simple_any setup_data_on_devices(const solver_type solver, const soa_matrix<real_type> &A) const final;
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix_explicit_impl
     */
    [[nodiscard]] ::plssvm::detail::simple_any assemble_kernel_matrix(const solver_type solver, const parameter &params, const ::plssvm::detail::simple_any &data, const std::vector<real_type> &q_red, const real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::blas_level_3
     */
    void blas_level_3(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const final;

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
     * @brief Synchronize the device denoted by @p queue.
     * @param[in] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(const queue_type &queue) const = 0;
    /**
     * @brief Return the maximum allowed work group size.
     * @return the maximum allowed work group size (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::size_t get_max_work_group_size() const = 0;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @brief Explicitly assemble the kernel matrix on the device.
     * @param[in] params the parameters (e.g., kernel function) used to assemble the kernel matrix
     * @param[in] data_d the data set to create the kernel matrix from
     * @param[in] q_red_d the vector used in the dimensional reduction
     * @param[in] QA_cost the scalar used in the dimensional reduction
     * @return the explicit kernel matrix stored on the device (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type &data_d, const device_ptr_type &q_red_d, real_type QA_cost) const = 0;
    /**
     * @brief Perform an explicit BLAS level 3 operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` matrix, @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    virtual void run_blas_level_3_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type &A, const device_ptr_type &B, real_type beta, device_ptr_type &C) const = 0;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @brief Calculate the `w` vector used to speedup the linear kernel prediction.
     * @param[in] alpha_d the support vector weights
     * @param[in] sv_d the support vectors
     * @return the `w` vector (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d) const = 0;
    /**
     * @brief Predict the values of the new @p predict_points_d using the previously learned weights @p alpha_d, biases @p rho_d, and support vectors @p sv_d.
     * @param[in] params the parameter used to predict the values (e.g., the used kernel function)
     * @param[in] w the `w` used to speedup the linear kernel prediction (only used in the linear kernel case)
     * @param[in] alpha_d the previously learned weights
     * @param[in] rho_d the previously calculated biases
     * @param[in] sv_d the previously learned support vectors
     * @param[in] predict_points_d the new data points to predict
     * @return the predicted values (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual device_ptr_type run_predict_kernel(const parameter &params, const device_ptr_type &w, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d) const = 0;

    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
};

//***************************************************//
//                        fit                        //
//***************************************************//
template <template <typename> typename device_ptr_t, typename queue_t>
::plssvm::detail::simple_any gpu_csvm<device_ptr_t, queue_t>::setup_data_on_devices(const solver_type solver, const soa_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices may not be empty!");
    PLSSVM_ASSERT(A.is_padded(), "Tha matrix to setup on the devices must be padded!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        // initialize the data on the device
        device_ptr_type data_d{ A.shape(), A.padding(), devices_[0] };
        data_d.copy_to_device(A);

        return ::plssvm::detail::simple_any{ std::move(data_d) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assembling the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
::plssvm::detail::simple_any gpu_csvm<device_ptr_t, queue_t>::assemble_kernel_matrix(const solver_type solver, const parameter &params, const ::plssvm::detail::simple_any &data, const std::vector<real_type> &q_red, real_type QA_cost) const {
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        // get the pointer to the data that already is on the device
        const device_ptr_type &data_d = data.get<device_ptr_type>();
        [[maybe_unused]] const std::size_t num_rows_reduced = data_d.size(0) - 1;
        [[maybe_unused]] const std::size_t num_features = data_d.size(1);

        PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be given!");
        PLSSVM_ASSERT(num_rows_reduced + PADDING_SIZE >= num_rows_reduced, "The number of rows with padding ({}) must be greater or equal to the number of rows without padding!", num_rows_reduced + PADDING_SIZE, num_rows_reduced);
        PLSSVM_ASSERT(num_features > 0, "At least one feature must be given!");
        PLSSVM_ASSERT((num_rows_reduced + PADDING_SIZE + 1) * (num_features + PADDING_SIZE) == data_d.size(),
                      "The number of values on the device data array is {}, but the provided sizes are {} ((num_rows_reduced + 1) * num_features)",
                      data_d.size(),
                      (num_rows_reduced + PADDING_SIZE + 1) * (num_features + PADDING_SIZE));

        // allocate memory for the values currently not on the device
        device_ptr_type q_red_d{ q_red.size() + PADDING_SIZE, devices_[0] };
        q_red_d.copy_to_device(q_red, 0, q_red.size());
        q_red_d.memset(0, q_red.size());
        device_ptr_type kernel_matrix = this->run_assemble_kernel_matrix_explicit(params, data_d, q_red_d, QA_cost);

#if defined(PLSSVM_USE_GEMM)
        PLSSVM_ASSERT((num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE) == kernel_matrix.size(),
                      "The kernel matrix must be a quadratic matrix with (num_rows_reduced + PADDING_SIZE)^2 ({}) entries, but is {}!",
                      (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE),
                      kernel_matrix.size());
#else
        PLSSVM_ASSERT((num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2 == kernel_matrix.size(),
                      "The kernel matrix must be a triangular matrix only with (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2 ({}) entries, but is {}!",
                      (num_rows_reduced + PADDING_SIZE) * (num_rows_reduced + PADDING_SIZE + 1) / 2,
                      kernel_matrix.size());
#endif

        return ::plssvm::detail::simple_any{ std::move(kernel_matrix) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assembling the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
void gpu_csvm<device_ptr_t, queue_t>::blas_level_3(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(B.is_padded(), "The B matrix must be padded!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix may not be empty!");
    PLSSVM_ASSERT(C.is_padded(), "The C matrix must be padded!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        const device_ptr_type &A_d = A.get<device_ptr_type>();
        PLSSVM_ASSERT(!A_d.empty(), "The A matrix may not be empty!");

        const std::size_t num_rhs = B.num_rows();
        const std::size_t num_rows = B.num_cols();

        // allocate memory on the device
        device_ptr_type B_d{ B.shape(), B.padding(), devices_[0] };
        B_d.copy_to_device(B);
        device_ptr_type C_d{ C.shape(), C.padding(), devices_[0] };
        C_d.copy_to_device(C);

        this->run_blas_level_3_kernel_explicit(num_rows, num_rhs, num_rows, alpha, A_d, B_d, beta, C_d);

        C_d.copy_to_host(C);
        C.restore_padding();
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("The GEMM calculation using the {} CG variation is currently not implemented!", solver) };
    }
}

//***************************************************//
//                   predict, score                  //
//***************************************************//
template <template <typename> typename device_ptr_t, typename queue_t>
aos_matrix<real_type> gpu_csvm<device_ptr_t, queue_t>::predict_values(const parameter &params,
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

    // defined sizes
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    device_ptr_type sv_d{ support_vectors.shape(), support_vectors.padding(), devices_[0] };
    sv_d.copy_to_device(support_vectors);
    device_ptr_type predict_points_d{ predict_points.shape(), predict_points.padding(), devices_[0] };
    predict_points_d.copy_to_device(predict_points);

    device_ptr_type w_d;  // only used when predicting linear kernel functions
    device_ptr_type alpha_d{ alpha.shape(), alpha.padding(), devices_[0] };
    alpha_d.copy_to_device(alpha);
    device_ptr_type rho_d{ num_classes + PADDING_SIZE, devices_[0] };
    rho_d.copy_to_device(rho, 0, rho.size());
    rho_d.memset(0, rho.size());

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w_d = this->run_w_kernel(alpha_d, sv_d);

            // convert 1D result to aos_matrix out-parameter
            w = soa_matrix<real_type>{ num_classes, num_features, PADDING_SIZE, PADDING_SIZE };
            w_d.copy_to_host(w);
            w.restore_padding();
        } else {
            // w already provided -> copy to device
            w_d = device_ptr_type{ { num_classes, num_features }, { PADDING_SIZE, PADDING_SIZE }, devices_[0] };
            w_d.copy_to_device(w);
        }
    }

    // predict
    const device_ptr_type out_d = this->run_predict_kernel(params, w_d, alpha_d, rho_d, sv_d, predict_points_d);

    // copy results back to host
    aos_matrix<real_type> out_ret{ num_predict_points, num_classes, PADDING_SIZE, PADDING_SIZE };
    out_d.copy_to_host(out_ret);
    out_ret.restore_padding();

    return out_ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_CSVM_HPP_
