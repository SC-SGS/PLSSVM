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

#include "plssvm/constants.hpp"                   // plssvm::real_type, plssvm::{THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE}
#include "plssvm/csvm.hpp"                        // plssvm::csvm
#include "plssvm/detail/logger.hpp"               // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/matrix.hpp"                      // plssvm::aos_matrix
#include "plssvm/parameter.hpp"                   // plssvm::parameter

#include "fmt/chrono.h"                           // output std::chrono times using {fmt}
#include "fmt/core.h"                             // fmt::format

#include <algorithm>                              // std::min, std::all_of, std::adjacent_find
#include <chrono>                                 // std::chrono::{milliseconds, steady_clock, duration_cast}
#include <cmath>                                  // std::ceil
#include <cstddef>                                // std::size_t
#include <functional>                             // std::less_equal
#include <iostream>                               // std::clog, std::cout, std::endl
#include <tuple>                                  // std::tuple, std::make_tuple
#include <utility>                                // std::forward, std::pair, std::move, std::make_pair
#include <vector>                                 // std::vector

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
    /**
     * @brief Returns the number of usable devices given the kernel function @p kernel and the number of features @p num_features.
     * @details Only the linear kernel supports multi-GPU execution, i.e., for the polynomial and rbf kernel, this function **always** returns 1.
     *          In addition, at most @p num_features devices may be used (i.e., if **more** devices than features are present not all devices are used).
     * @param[in] kernel the kernel function type
     * @param[in] num_features the number of features
     * @return the number of usable devices; may be less than the discovered devices in the system (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t select_num_used_devices(kernel_function_type kernel, std::size_t num_features) const noexcept;
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each device.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduced data
     */
    void device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) const;

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
    [[nodiscard]] ::plssvm::detail::simple_any assemble_kernel_matrix(const solver_type solver, const parameter &params, const ::plssvm::detail::simple_any & data, const std::vector<real_type> &q_red, const real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::kernel_matrix_matmul_explicit
     */
    void blas_gemm(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, aos_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const final;

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
    [[nodiscard]] virtual device_ptr_type run_assemble_kernel_matrix_explicit(const parameter &params, const device_ptr_type & data_d, const device_ptr_type &q_red_d, real_type QA_cost) const = 0;
    /**
     * @brief Perform an explicit BLAS GEMM operation: `C = alpha * A * B + beta * C` where @p A is a `m x k` matrix, @p B is a `k x n` matrix, @p C is a `m x n` matrix, and @p alpha and @p beta are scalars.
     * @param[in] m the number of rows in @p A and @p C
     * @param[in] n the number of columns in @p B and @p C
     * @param[in] k the number of rows in @p A and number of columns in @p B
     * @param[in] alpha the scalar alpha value
     * @param[in] A the matrix @p A
     * @param[in] B the matrix @p B
     * @param[in] beta the scalar beta value
     * @param[in,out] C the matrix @p C, also used as result matrix
     */
    virtual void run_gemm_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type &A, const device_ptr_type &B, real_type beta, device_ptr_type &C) const = 0;

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

template <template <typename> typename device_ptr_t, typename queue_t>
std::size_t gpu_csvm<device_ptr_t, queue_t>::select_num_used_devices(const kernel_function_type, const std::size_t) const noexcept {
//    PLSSVM_ASSERT(num_features > 0, "At lest one feature must be given!");
//
//    // polynomial and rbf kernel currently only support single GPU execution
//    if ((kernel == kernel_function_type::polynomial || kernel == kernel_function_type::rbf) && devices_.size() > 1) {
//        std::clog << fmt::format("Warning: found {} devices, however only 1 device can be used since the polynomial and rbf kernels currently only support single GPU execution!", devices_.size()) << std::endl;
//        return 1;
//    }
//
//    // the number of used devices may not exceed the number of features
//    const std::size_t num_used_devices = std::min(devices_.size(), num_features);
//    if (num_used_devices < devices_.size()) {
//        std::clog << fmt::format("Warning: found {} devices, however only {} device(s) can be used since the data set only has {} features!", devices_.size(), num_used_devices, num_features) << std::endl;
//    }
//    return num_used_devices;
    // TODO: currently only a single device is supported!
    return 1;
}

template <template <typename> typename device_ptr_t, typename queue_t>
void gpu_csvm<device_ptr_t, queue_t>::device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) const {
    PLSSVM_ASSERT(!buffer_d.empty(), "The buffer_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(buffer_d.cbegin(), buffer_d.cend(), [](const device_ptr_type &ptr) { return !ptr.empty(); }), "Each device_ptr in buffer_d must at least contain one data point!");
    PLSSVM_ASSERT(!buffer.empty(), "The buffer array may not be empty!");

    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].copy_to_host(buffer, 0, buffer.size());

    if (buffer_d.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<device_ptr_type>::size_type device = 1; device < buffer_d.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].copy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for default(none) shared(buffer_d, buffer)
        for (typename std::vector<device_ptr_type>::size_type device = 0; device < buffer_d.size(); ++device) {
            buffer_d[device].copy_to_device(buffer, 0, buffer.size());
        }
    }
}

//***************************************************//
//                        fit                        //
//***************************************************//
template <template <typename> typename device_ptr_t, typename queue_t>
::plssvm::detail::simple_any gpu_csvm<device_ptr_t, queue_t>::setup_data_on_devices(const solver_type solver, const soa_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        // initialize the data on the device
        device_ptr_type data_d{ A.shape(), devices_[0] };  // TODO: don't copy last data point to device?
        data_d.copy_to_device(A.data());

        return ::plssvm::detail::simple_any{ std::move(data_d) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
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
        PLSSVM_ASSERT(num_features > 0, "At least one feature must be given!");
        PLSSVM_ASSERT((num_rows_reduced + 1) * num_features == data_d.size(),
                      "The number of values on the device data array is {}, but the provided sizes are {} ((num_rows_reduced + 1) * num_features)",
                      data_d.size(), (num_rows_reduced + 1) * num_features);

        // allocate memory for the values currently not on the device
        device_ptr_type q_red_d{ q_red.size(), devices_[0] };
        q_red_d.copy_to_device(q_red);
        device_ptr_type kernel_matrix = this->run_assemble_kernel_matrix_explicit(params, data_d, q_red_d, QA_cost);

        PLSSVM_ASSERT(num_rows_reduced * (num_rows_reduced + 1) / 2 == kernel_matrix.size(),
                      "The kernel matrix must only save one triangular matrix (symmetric) with {} entries, but is {}!",
                      num_rows_reduced * (num_rows_reduced + 1) / 2, kernel_matrix.size());

        return ::plssvm::detail::simple_any{ std::move(kernel_matrix) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
void gpu_csvm<device_ptr_t, queue_t>::blas_gemm(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const soa_matrix<real_type> &B, const real_type beta, soa_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        const device_ptr_type &A_d = A.get<device_ptr_type>();
        PLSSVM_ASSERT(!A_d.empty(), "The A matrix may not be empty!");

        const std::size_t num_rhs = B.num_rows();
        const std::size_t num_rows = B.num_cols();

        // allocate memory on the device
        static device_ptr_type B_d{ B.shape(), devices_[0] };
        if (B_d.size() != B.num_entries()) {
            B_d = device_ptr_type{ B.shape(), devices_[0] };
        }
        B_d.copy_to_device(B.data());
        static device_ptr_type C_d{ C.shape(), devices_[0] };
        if (C_d.size() != C.num_entries()) {
            C_d = device_ptr_type{ C.shape(), devices_[0] };
        }
        C_d.copy_to_device(C.data());

        this->run_gemm_kernel_explicit(num_rows, num_rhs, num_rows, alpha, A_d, B_d, beta, C_d);

        C_d.copy_to_host(C.data());
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
                                                                                   aos_matrix<real_type> &w,
                                                                                   const soa_matrix<real_type> &predict_points) const {
    PLSSVM_ASSERT(!support_vectors.empty(), "The support vectors must not be empty!");
    PLSSVM_ASSERT(!alpha.empty(), "The alpha vectors (weights) must not be empty!");
    PLSSVM_ASSERT(support_vectors.num_rows() == alpha.num_cols(), "The number of support vectors ({}) and number of weights ({}) must be the same!", support_vectors.num_rows(), alpha.num_cols());
    PLSSVM_ASSERT(rho.size() == alpha.num_rows(), "The number of rho values ({}) and the number of weights ({}) must be the same!", rho.size(), alpha.num_rows());
    PLSSVM_ASSERT(w.empty() || support_vectors.num_cols() == w.num_cols(), "Either w must be empty or contain exactly the same number of values as features are present ({})!", support_vectors.num_cols());
    PLSSVM_ASSERT(w.empty() || alpha.num_rows() == w.num_rows(), "Either w must be empty or contain exactly the same number of vectors ({}) as the alpha vector ({})!", w.num_rows(), alpha.num_rows());
    PLSSVM_ASSERT(!predict_points.empty(), "The data points to predict must not be empty!");
    PLSSVM_ASSERT(support_vectors.num_cols() == predict_points.num_cols(), "The number of features in the support vectors ({}) must be the same as in the data points to predict ({})!", support_vectors.num_cols(), predict_points.num_cols());

    using namespace plssvm::operators;

    // defined sizes
    const std::size_t num_classes = alpha.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    device_ptr_type sv_d{ support_vectors.shape(), devices_[0] };
    sv_d.copy_to_device(support_vectors.data());
    device_ptr_type predict_points_d{ predict_points.shape(), devices_[0] };
    predict_points_d.copy_to_device(predict_points.data());

    device_ptr_type w_d;  // only used when predicting linear kernel functions
    device_ptr_type alpha_d{ alpha.shape(), devices_[0] };
    alpha_d.copy_to_device(alpha.data());
    device_ptr_type rho_d{ num_classes, devices_[0] };
    rho_d.copy_to_device(rho);

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w_d = run_w_kernel(alpha_d, sv_d);

            // convert 1D result to aos_matrix out-parameter
            w = aos_matrix<real_type>{ num_classes, num_features };
            w_d.copy_to_host(w.data());
        } else {
            // w already provided -> copy to device
            w_d = device_ptr_type{ { num_classes, num_features }, devices_[0] };
            w_d.copy_to_device(w.data());
        }
    }

    // predict
    const device_ptr_type out_d = run_predict_kernel(params, w_d, alpha_d, rho_d, sv_d, predict_points_d);;

    // copy results back to host
    aos_matrix<real_type> out_ret{ num_predict_points, num_classes };
    out_d.copy_to_host(out_ret.data());
    return out_ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_CSVM_HPP_
