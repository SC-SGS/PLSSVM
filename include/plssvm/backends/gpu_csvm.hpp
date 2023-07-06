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
#include "plssvm/detail/execution_range.hpp"      // plssvm::detail::execution_range
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
    explicit gpu_csvm(plssvm::parameter params = {}) :
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
    [[nodiscard]] ::plssvm::detail::simple_any setup_data_on_devices(const solver_type solver, const aos_matrix<real_type> &A) const final;

    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix_explicit_impl
     */
    [[nodiscard]] ::plssvm::detail::simple_any assemble_kernel_matrix(const solver_type solver, const ::plssvm::detail::parameter<real_type> &params, const ::plssvm::detail::simple_any & data, const std::vector<real_type> &q_red, const real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::kernel_matrix_matmul_explicit
     */
    void blas_gemm(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<real_type> predict_values(const parameter<real_type> &params, const aos_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, aos_matrix<real_type> &w, const aos_matrix<real_type> &predict_points) const final;

    /**
     * @brief Precalculate the `w` vector to speedup up the prediction using the linear kernel function.
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] data_last_d the last data point of the data set located on the device(s)
     * @param[in] alpha_d the previously learned weights located on the device(s)
     * @param[in] num_data_points the number of data points in @p data_p
     * @param[in] feature_ranges the range of features a specific device is responsible for
     * @return the `w` vector (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<device_ptr_type> &data_d, const std::vector<device_ptr_type> &data_last_d, const std::vector<device_ptr_type> &alpha_d, std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges) const;

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the device denoted by @p device.
     * @param[in] device the device ID denoting the device on which the kernel should be executed
     * @param[in] params the SVM parameter used (e.g., kernel_type)
     * @param[in] q_d subvector of the least-squares matrix equation located on the device(s)
     * @param[in,out] r_d the result vector located on the device(s)
     * @param[in] x_d the right-hand side of the equation located on the device(s)
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] feature_ranges the range of features a specific device is responsible for
     * @param[in] QA_cost a value used in the dimensional reduction
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] dept the number of data points after the dimensional reduction
     * @param[in] boundary_size the size of the padding boundary
     */
    void run_device_kernel(std::size_t device, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, const std::vector<std::size_t> &feature_ranges, real_type QA_cost, real_type add, std::size_t dept, std::size_t boundary_size) const;

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    // Note: there are two versions of each function (one for float and one for double) since virtual template functions are not allowed in C++!

    /**
     * @brief Synchronize the device denoted by @p queue.
     * @param[in] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(const queue_type &queue) const = 0;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    virtual device_ptr_type run_assemble_kernel_matrix_explicit(const ::plssvm::detail::parameter<real_type> &params, const device_ptr_type & data_d, const device_ptr_type &q_red_d, real_type QA_cost) const = 0;

    virtual void run_gemm_kernel_explicit(std::size_t m, std::size_t n, std::size_t k, real_type alpha, const device_ptr_type &A, const device_ptr_type &B, real_type beta, device_ptr_type &C) const = 0;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    virtual device_ptr_type run_predict_kernel(const parameter<real_type> &params, const device_ptr_type &w, const device_ptr_type &alpha_d, const device_ptr_type &rho_d, const device_ptr_type &sv_d, const device_ptr_type &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const = 0;

    virtual device_ptr_type run_w_kernel(const device_ptr_type &alpha_d, const device_ptr_type &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const = 0;


    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
};

template <template <typename> typename device_ptr_t, typename queue_t>
std::size_t gpu_csvm<device_ptr_t, queue_t>::select_num_used_devices(const kernel_function_type kernel, const std::size_t num_features) const noexcept {
    PLSSVM_ASSERT(num_features > 0, "At lest one feature must be given!");

    // polynomial and rbf kernel currently only support single GPU execution
    if ((kernel == kernel_function_type::polynomial || kernel == kernel_function_type::rbf) && devices_.size() > 1) {
        std::clog << fmt::format("Warning: found {} devices, however only 1 device can be used since the polynomial and rbf kernels currently only support single GPU execution!", devices_.size()) << std::endl;
        return 1;
    }

    // the number of used devices may not exceed the number of features
    const std::size_t num_used_devices = std::min(devices_.size(), num_features);
    if (num_used_devices < devices_.size()) {
        std::clog << fmt::format("Warning: found {} devices, however only {} device(s) can be used since the data set only has {} features!", devices_.size(), num_used_devices, num_features) << std::endl;
    }
    return num_used_devices;
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
::plssvm::detail::simple_any gpu_csvm<device_ptr_t, queue_t>::setup_data_on_devices(const solver_type solver, const aos_matrix<real_type> &A) const {
    PLSSVM_ASSERT(!A.empty(), "The matrix to setup on the devices may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        // initialize the data on the device
        device_ptr_type data_d{ A.shape() };  // TODO: don't copy last data point to device?
        data_d.copy_to_device(A.data());

        return ::plssvm::detail::simple_any{ std::move(data_d) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
::plssvm::detail::simple_any gpu_csvm<device_ptr_t, queue_t>::assemble_kernel_matrix(const solver_type solver, const ::plssvm::detail::parameter<real_type> &params, const ::plssvm::detail::simple_any &data, const std::vector<real_type> &q_red, real_type QA_cost) const {
    PLSSVM_ASSERT(!q_red.empty(), "The q_red vector may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        // get the pointer to the data that already is on the device
        const device_ptr_type &data_d = data.get<device_ptr_type>();
        const std::size_t num_rows_reduced = data_d.size(0) - 1;
        const std::size_t num_features = data_d.size(1);

        PLSSVM_ASSERT(num_rows_reduced > 0, "At least one row must be given!");
        PLSSVM_ASSERT(num_features > 0, "At least one feature must be given!");
        PLSSVM_ASSERT((num_rows_reduced + 1) * num_features == data_d.size(),
                      "The number of values on the device data array is {}, but the provided sizes are {} ((num_rows_reduced + 1) * num_features)",
                      data_d.size(), (num_rows_reduced + 1) * num_features);

        // allocate memory for the values currently not on the device
        device_ptr_type q_red_d{ q_red.size() };
        q_red_d.copy_to_device(q_red);
        device_ptr_type kernel_matrix = this->run_assemble_kernel_matrix_explicit(params, data_d, q_red_d, QA_cost);

        PLSSVM_ASSERT(num_rows_reduced * num_rows_reduced == kernel_matrix.size(),
                      "The kernel matrix must be a quadratic matrix with num_rows_reduced^2 ({}) entries, but is {}!",
                      num_rows_reduced * num_rows_reduced, kernel_matrix.size());

        return ::plssvm::detail::simple_any{ std::move(kernel_matrix) };
    } else {
        // TODO: implement for other solver types
        throw exception{ fmt::format("Assemblying the kernel matrix using the {} CG variation is currently not implemented!", solver) };
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
void gpu_csvm<device_ptr_t, queue_t>::blas_gemm(const solver_type solver, const real_type alpha, const ::plssvm::detail::simple_any &A, const aos_matrix<real_type> &B, const real_type beta, aos_matrix<real_type> &C) const {
    PLSSVM_ASSERT(!B.empty(), "The B matrix may not be empty!");
    PLSSVM_ASSERT(!C.empty(), "The C matrix may not be empty!");
    PLSSVM_ASSERT(solver != solver_type::automatic, "An explicit solver type must be provided instead of solver_type::automatic!");

    if (solver == solver_type::cg_explicit) {
        const device_ptr_type &A_d = A.get<device_ptr_type>();
        PLSSVM_ASSERT(!A_d.empty(), "The A matrix may not be empty!");

        const std::size_t num_rhs = B.num_rows();
        const std::size_t num_rows = B.num_cols();

        // allocate memory on the device
        static device_ptr_type B_d{ B.shape() };
        if (B_d.size() != B.num_entries()) {
            B_d = device_ptr_type{ B.shape() };
        }
        B_d.copy_to_device(B.data());
        static device_ptr_type C_d{ C.shape() };
        if (C_d.size() != C.num_entries()) {
            C_d = device_ptr_type{ C.shape() };
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
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::calculate_w(const std::vector<device_ptr_type> &data_d,
                                                                    const std::vector<device_ptr_type> &data_last_d,
                                                                    const std::vector<device_ptr_type> &alpha_d,
                                                                    const std::size_t num_data_points,
                                                                    const std::vector<std::size_t> &feature_ranges) const {
    PLSSVM_ASSERT(!data_d.empty(), "The data_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_d.cbegin(), data_d.cend(), [](const device_ptr_type &ptr) { return !ptr.empty(); }), "Each device_ptr in data_d must at least contain one data point!");
    PLSSVM_ASSERT(!data_last_d.empty(), "The data_last_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_last_d.cbegin(), data_last_d.cend(), [](const device_ptr_type &ptr) { return !ptr.empty(); }), "Each device_ptr in data_last_d must at least contain one data point!");
    PLSSVM_ASSERT(data_d.size() == data_last_d.size(), "The number of used devices to the data_d and data_last_d vectors must be equal!: {} != {}", data_d.size(), data_last_d.size());
    PLSSVM_ASSERT(!alpha_d.empty(), "The alpha_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(alpha_d.cbegin(), alpha_d.cend(), [](const device_ptr_type &ptr) { return !ptr.empty(); }), "Each device_ptr in alpha_d must at least contain one data point!");
    PLSSVM_ASSERT(data_d.size() == alpha_d.size(), "The number of used devices to the data_d and alpha_d vectors must be equal!: {} != {}", data_d.size(), alpha_d.size());
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point must be used to calculate q!");
    PLSSVM_ASSERT(feature_ranges.size() == data_d.size() + 1, "The number of values in the feature_range vector must be exactly one more than the number of used devices!: {} != {} + 1", feature_ranges.size(), data_d.size());
    PLSSVM_ASSERT(std::adjacent_find(feature_ranges.cbegin(), feature_ranges.cend(), std::less_equal<>{}) != feature_ranges.cend(), "The feature ranges are not monotonically increasing!");

    const std::size_t num_used_devices = data_d.size();

    // create w vector and fill with zeros
    std::vector<real_type> w(feature_ranges.back(), real_type{ 0.0 });

    #pragma omp parallel for default(none) shared(num_used_devices, devices_, feature_ranges, alpha_d, data_d, data_last_d, w) firstprivate(num_data_points, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        // feature splitting on multiple devices
        const std::size_t num_features_in_range = feature_ranges[device + 1] - feature_ranges[device];

        // create the w vector on the device
        device_ptr_type w_d = device_ptr_type{ num_features_in_range, devices_[device] };

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features_in_range) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_features_in_range) });

        // calculate the w vector on the device
        run_w_kernel(device, range, w_d, alpha_d[device], data_d[device], data_last_d[device], num_data_points, num_features_in_range);
        device_synchronize(devices_[device]);

        // copy back to host memory
        w_d.copy_to_host(w.data() + feature_ranges[device], 0, num_features_in_range);
    }
    return w;
}

template <template <typename> typename device_ptr_t, typename queue_t>
void gpu_csvm<device_ptr_t, queue_t>::run_device_kernel(const std::size_t device, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, const std::vector<std::size_t> &feature_ranges, const real_type QA_cost, const real_type add, const std::size_t dept, const std::size_t boundary_size) const {
    PLSSVM_ASSERT(device < devices_.size(), "Requested device {}, but only {} device(s) are available!", device, devices_.size());
    PLSSVM_ASSERT(!q_d.empty(), "The q_d device_ptr may not be empty!");
    PLSSVM_ASSERT(!r_d.empty(), "The r_d device_ptr may not be empty!");
    PLSSVM_ASSERT(!x_d.empty(), "The x_d device_ptr may not be empty!");
    PLSSVM_ASSERT(!data_d.empty(), "The data_d device_ptr may not be empty!");
    PLSSVM_ASSERT(std::adjacent_find(feature_ranges.cbegin(), feature_ranges.cend(), std::less_equal<>{}) != feature_ranges.cend(), "The feature ranges are not monotonically increasing!");
    PLSSVM_ASSERT(add == real_type{ -1.0 } || add == real_type{ 1.0 }, "add must either by -1.0 or 1.0, but is {}!", add);
    PLSSVM_ASSERT(dept > 0, "At least one data point must be used to calculate q!");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept) / static_cast<real_type>(boundary_size)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel(device, range, params, q_d, r_d, x_d, data_d, QA_cost, add, dept + boundary_size, feature_ranges[device + 1] - feature_ranges[device]);
}


template <template <typename> typename device_ptr_t, typename queue_t>
aos_matrix<real_type> gpu_csvm<device_ptr_t, queue_t>::predict_values(const parameter<real_type> &params,
                                                                                   const aos_matrix<real_type> &support_vectors,
                                                                                   const aos_matrix<real_type> &alpha,
                                                                                   const std::vector<real_type> &rho,
                                                                                   aos_matrix<real_type> &w,
                                                                                   const aos_matrix<real_type> &predict_points) const {
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
    const std::size_t num_support_vectors = support_vectors.num_rows();
    const std::size_t num_predict_points = predict_points.num_rows();
    const std::size_t num_features = predict_points.num_cols();

    device_ptr_type sv_d{ num_support_vectors * num_features };
    sv_d.copy_to_device(support_vectors.data());
    device_ptr_type predict_points_d{ num_predict_points * num_features };
    predict_points_d.copy_to_device(predict_points.data());

    device_ptr_type w_d;  // only used when predicting linear kernel functions
    device_ptr_type alpha_d{ num_classes * num_support_vectors };
    alpha_d.copy_to_device(alpha.data());
    device_ptr_type rho_d{ num_classes };
    rho_d.copy_to_device(rho);

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w_d = run_w_kernel(alpha_d, sv_d, num_classes, num_support_vectors, num_features);

            // convert 1D result to aos_matrix out-parameter
            w = aos_matrix<real_type>{ num_classes, num_features };
            w_d.copy_to_host(w.data());
        } else {
            // w already provided -> copy to device
            w_d = device_ptr_type{ num_classes * num_features };
            w_d.copy_to_device(w.data());
        }
    }

    // predict
    const device_ptr_type out_d = run_predict_kernel(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_support_vectors, num_predict_points, num_features);;

    // copy results back to host
    aos_matrix<real_type> out_ret{ num_predict_points, num_classes };
    out_d.copy_to_host(out_ret.data());
    return out_ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_CSVM_HPP_
