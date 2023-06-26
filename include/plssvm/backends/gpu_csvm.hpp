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

#include "plssvm/constants.hpp"                   // plssvm::{THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE}
#include "plssvm/csvm.hpp"                        // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"      // plssvm::detail::execution_range
#include "plssvm/detail/layout.hpp"               // plssvm::detail::{transform_to_layout, layout_type}
#include "plssvm/detail/logger.hpp"               // plssvm::detail::log, plssvm::verbosity_level
#include "plssvm/detail/performance_tracker.hpp"  // plssvm::detail::tracking_entry, PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/parameter.hpp"                   // plssvm::parameter
#include "plssvm/detail/matrix.hpp"               // plssvm::detail::aos_matrix

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
    template <typename real_type>
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
     * @copydoc plssvm::csvm::solve_system_of_linear_equations
     */
    [[nodiscard]] std::pair<detail::aos_matrix<float>, std::vector<float>> solve_system_of_linear_equations(const parameter<float> &params, const detail::aos_matrix<float> &A, detail::aos_matrix<float> B, float eps, unsigned long long max_iter) const final { return this->solve_system_of_linear_equations_impl(params, A, std::move(B), eps, max_iter); }
    /**
     * @copydoc plssvm::csvm::solve_system_of_linear_equations
     */
    [[nodiscard]] std::pair<detail::aos_matrix<double>, std::vector<double>> solve_system_of_linear_equations(const parameter<double> &params, const detail::aos_matrix<double> &A, detail::aos_matrix<double> B, double eps, unsigned long long max_iter) const final { return this->solve_system_of_linear_equations_impl(params, A, std::move(B), eps, max_iter); }
    /**
     * @copydoc plssvm::csvm::solve_system_of_linear_equations
     */
    template <typename real_type>
    [[nodiscard]] std::pair<detail::aos_matrix<real_type>, std::vector<real_type>> solve_system_of_linear_equations_impl(const parameter<real_type> &params, const detail::aos_matrix<real_type> &A, detail::aos_matrix<real_type> B, real_type eps, unsigned long long max_iter) const;

    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] detail::aos_matrix<float> predict_values(const parameter<float> &params, const detail::aos_matrix<float> &support_vectors, const detail::aos_matrix<float> &alpha, const std::vector<float> &rho, detail::aos_matrix<float> &w, const detail::aos_matrix<float> &predict_points) const final { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] detail::aos_matrix<double> predict_values(const parameter<double> &params, const detail::aos_matrix<double> &support_vectors, const detail::aos_matrix<double> &alpha, const std::vector<double> &rho, detail::aos_matrix<double> &w, const detail::aos_matrix<double> &predict_points) const final { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    template <typename real_type>
    [[nodiscard]] detail::aos_matrix<real_type> predict_values_impl(const parameter<real_type> &params, const detail::aos_matrix<real_type> &support_vectors, const detail::aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, detail::aos_matrix<real_type> &w, const detail::aos_matrix<real_type> &predict_points) const;

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
     * @brief Performs all necessary steps such that the data is available on the device with the correct layout.
     * @details Distributed the data evenly across all devices, adds padding data points, and transforms the data layout to SoA.
     * @tparam real_type the type of the data points (either `float` or `double`)
     * @param[in] data the data that should be copied to the device(s)
     * @param[in] num_data_points_to_setup the number of data points that should be copied to the device
     * @param[in] num_features_to_setup the number of features in the data set
     * @param[in] boundary_size the size of the padding boundary
     * @param[in] num_used_devices the number of devices to distribute the data across
     * @return a tuple: [pointers to the main data distributed across the devices, pointers to the last data point of the data set distributed across the devices, the feature ranges a specific device is responsible for] (`[[nodiscard]]`)
     */
    template <typename real_type>
    [[nodiscard]] std::tuple<std::vector<device_ptr_type<real_type>>, std::vector<device_ptr_type<real_type>>, std::vector<std::size_t>> setup_data_on_device(const detail::aos_matrix<real_type> &data, std::size_t num_data_points_to_setup, std::size_t num_features_to_setup, std::size_t boundary_size, std::size_t num_used_devices) const;

    /**
     * @brief Calculate the `q` vector used in the dimensional reduction.
     * @tparam real_type the type of the data points (either `float` or `double`)
     * @param[in] params the SVM parameter used to calculate `q` (e.g., kernel_type)
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] data_last_d the last data point of the data set located on the device(s)
     * @param[in] num_data_points the number of data points in @p data_p
     * @param[in] feature_ranges the range of features a specific device is responsible for
     * @param[in] boundary_size the size of the padding boundary
     * @return the `q` vector (`[[nodiscard]]`)
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> generate_q(const parameter<real_type> &params, const std::vector<device_ptr_type<real_type>> &data_d, const std::vector<device_ptr_type<real_type>> &data_last_d, std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges, std::size_t boundary_size) const;
    /**
     * @brief Precalculate the `w` vector to speedup up the prediction using the linear kernel function.
     * @tparam real_type the type of the data points (either `float` or `double`)
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] data_last_d the last data point of the data set located on the device(s)
     * @param[in] alpha_d the previously learned weights located on the device(s)
     * @param[in] num_data_points the number of data points in @p data_p
     * @param[in] feature_ranges the range of features a specific device is responsible for
     * @return the `w` vector (`[[nodiscard]]`)
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<device_ptr_type<real_type>> &data_d, const std::vector<device_ptr_type<real_type>> &data_last_d, const std::vector<device_ptr_type<real_type>> &alpha_d, std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges) const;

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
    template <typename real_type>
    void run_device_kernel(std::size_t device, const parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const std::vector<std::size_t> &feature_ranges, real_type QA_cost, real_type add, std::size_t dept, std::size_t boundary_size) const;
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each device.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduced data
     */
    template <typename real_type>
    void device_reduction(std::vector<device_ptr_type<real_type>> &buffer_d, std::vector<real_type> &buffer) const;

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
     * @brief Run the device kernel filling the `q` vector.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] params the SVM parameter used (e.g., kernel_type)
     * @param[out] q_d the `q` vector to fill located on the device(s)
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] data_last_d the last data point of the data set located on the device(s)
     * @param[in] num_data_points_padded the number of data points after the padding has been applied
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_q_kernel(std::size_t device, const detail::execution_range &range, const parameter<float> &params, device_ptr_type<float> &q_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_q_kernel
     */
    virtual void run_q_kernel(std::size_t device, const detail::execution_range &range, const parameter<double> &params, device_ptr_type<double> &q_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    /**
     * @brief Run the main device kernel used in the CG algorithm.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] params the SVM parameter used (e.g., kernel_type)
     * @param[in] q_d the `q` vector located on the device(s)
     * @param[in,out] r_d the result vector located on the device(s)
     * @param[in] x_d the right-hand side of the equation located on the device(s)
     * @param[in] data_d the data points used in the dimensional reduction located on the device(s)
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] QA_cost a value used in the dimensional reduction
     * @param[in] num_data_points_padded the number of data points after the padding has been applied
     * @param[in] num_features number of features used for the calculation in the @p device
     */
    virtual void run_svm_kernel(std::size_t device, const detail::execution_range &range, const parameter<float> &params, const device_ptr_type<float> &q_d, device_ptr_type<float> &r_d, const device_ptr_type<float> &x_d, const device_ptr_type<float> &data_d, float QA_cost, float add, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    /**
     * @copydoc plssvm::detail::gpu_csvm::run_svm_kernel
     */
    virtual void run_svm_kernel(std::size_t device, const detail::execution_range &range, const parameter<double> &params, const device_ptr_type<double> &q_d, device_ptr_type<double> &r_d, const device_ptr_type<double> &x_d, const device_ptr_type<double> &data_d, double QA_cost, double add, std::size_t num_data_points_padded, std::size_t num_features) const = 0;

    virtual device_ptr_type<float> run_predict_kernel(const parameter<float> &params, const device_ptr_type<float> &w, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &rho_d, const device_ptr_type<float> &sv_d, const device_ptr_type<float> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const = 0;
    virtual device_ptr_type<double> run_predict_kernel(const parameter<double> &params, const device_ptr_type<double> &w, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &rho_d, const device_ptr_type<double> &sv_d, const device_ptr_type<double> &predict_points_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_predict_points, std::size_t num_features) const = 0;

    virtual device_ptr_type<float> run_w_kernel(const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const = 0;
    virtual device_ptr_type<double> run_w_kernel(const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &sv_d, std::size_t num_classes, std::size_t num_sv, std::size_t num_features) const = 0;

    virtual device_ptr_type<float> assemble_kernel_matrix(const detail::parameter<float> &params, const device_ptr_type<float> &data_d, const std::vector<float> &q, float QA_cost, const std::size_t num_data_points, const std::size_t num_features) const = 0;
    virtual device_ptr_type<double> assemble_kernel_matrix(const detail::parameter<double> &params, const device_ptr_type<double> &data_d, const std::vector<double> &q, double QA_cost, const std::size_t num_data_points, const std::size_t num_features) const = 0;

    virtual void blas_gemm(std::size_t m, std::size_t n, std::size_t k, float alpha, const device_ptr_type<float> &A, const device_ptr_type<float> &B, float beta, device_ptr_type<float> &C) const = 0;
    virtual void blas_gemm(std::size_t m, std::size_t n, std::size_t k, double alpha, const device_ptr_type<double> &A, const device_ptr_type<double> &B, double beta, device_ptr_type<double> &C) const = 0;

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

/// @cond Doxygen_suppress
template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::tuple<std::vector<device_ptr_t<real_type>>, std::vector<device_ptr_t<real_type>>, std::vector<std::size_t>>
gpu_csvm<device_ptr_t, queue_t>::setup_data_on_device(const detail::aos_matrix<real_type> &data,
                                                      const std::size_t num_data_points_to_setup,
                                                      const std::size_t num_features_to_setup,
                                                      const std::size_t boundary_size,
                                                      const std::size_t num_used_devices) const {
    PLSSVM_ASSERT(!data.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(num_data_points_to_setup > 0, "At least one data point must be copied to the device!");
    PLSSVM_ASSERT(num_data_points_to_setup <= data.num_rows(), "Can't copy more data points to the device than are present!: {} <= {}", num_data_points_to_setup, data.num_rows());
    PLSSVM_ASSERT(num_features_to_setup > 0, "At least one feature must be copied to the device!");
    PLSSVM_ASSERT(num_features_to_setup <= data.num_cols(), "Can't copy more features to the device than are present!: {} <= {}", num_features_to_setup, data.num_cols());
    PLSSVM_ASSERT(num_used_devices <= devices_.size(), "Can't use more devices than are available!: {} <= {}", num_used_devices, devices_.size());

    // calculate the number of features per device
    std::vector<std::size_t> feature_ranges(num_used_devices + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= num_used_devices; ++device) {
        feature_ranges[device] = device * num_features_to_setup / num_used_devices;
    }

    // transform 2D to 1D AoS data
    // TODO: boundary?!
//    const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::aos, data, boundary_size, num_data_points_to_setup);

    std::vector<device_ptr_type<real_type>> data_last_d(num_used_devices);
    std::vector<device_ptr_type<real_type>> data_d(num_used_devices);

    #pragma omp parallel for default(none) shared(num_used_devices, devices_, feature_ranges, data_last_d, data_d, data) firstprivate(num_data_points_to_setup, boundary_size, num_features_to_setup)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        const std::size_t num_features_in_range = feature_ranges[device + 1] - feature_ranges[device];

        // initialize data_last on device
        data_last_d[device] = device_ptr_type<real_type>{ num_features_in_range + boundary_size, devices_[device] };
        data_last_d[device].memset(0);
        std::vector<real_type> back_data(num_features_in_range);
        for (std::size_t i = 0; i < num_features_in_range; ++i) {
            back_data[i] = data(data.num_rows() - 1, feature_ranges[device] + i);
        }
        data_last_d[device].copy_to_device(back_data);

        const std::size_t device_data_size = num_features_in_range * (num_data_points_to_setup + boundary_size);
        data_d[device] = device_ptr_type<real_type>{ device_data_size, devices_[device] };
        data_d[device].copy_to_device(data.data() + feature_ranges[device] * (num_data_points_to_setup + boundary_size), 0, device_data_size);
    }

    return std::make_tuple(std::move(data_d), std::move(data_last_d), std::move(feature_ranges));
}
/// @endcond

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::generate_q(const parameter<real_type> &params,
                                                                   const std::vector<device_ptr_type<real_type>> &data_d,
                                                                   const std::vector<device_ptr_type<real_type>> &data_last_d,
                                                                   const std::size_t num_data_points,
                                                                   const std::vector<std::size_t> &feature_ranges,
                                                                   const std::size_t boundary_size) const {
    PLSSVM_ASSERT(!data_d.empty(), "The data_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_d.cbegin(), data_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in data_d must at least contain one data point!");
    PLSSVM_ASSERT(!data_last_d.empty(), "The data_last_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_last_d.cbegin(), data_last_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in data_last_d must at least contain one data point!");
    PLSSVM_ASSERT(data_d.size() == data_last_d.size(), "The number of used devices to the data_d and data_last_d vectors must be equal!: {} != {}", data_d.size(), data_last_d.size());
    PLSSVM_ASSERT(num_data_points > 0, "At least one data point must be used to calculate q!");
    PLSSVM_ASSERT(feature_ranges.size() == data_d.size() + 1, "The number of values in the feature_range vector must be exactly one more than the number of used devices!: {} != {} + 1", feature_ranges.size(), data_d.size());
    PLSSVM_ASSERT(std::adjacent_find(feature_ranges.cbegin(), feature_ranges.cend(), std::less_equal<>{}) != feature_ranges.cend(), "The feature ranges are not monotonically increasing!");

    const std::size_t num_used_devices = data_d.size();
    std::vector<device_ptr_type<real_type>> q_d(num_used_devices);

    #pragma omp parallel for default(none) shared(num_used_devices, q_d, devices_, data_d, data_last_d, feature_ranges, params) firstprivate(num_data_points, boundary_size, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        q_d[device] = device_ptr_type<real_type>{ num_data_points + boundary_size, devices_[device] };
        q_d[device].memset(0);

        // feature splitting on multiple devices
        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_data_points) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_data_points) });

        run_q_kernel(device, range, params, q_d[device], data_d[device], data_last_d[device], num_data_points + boundary_size, feature_ranges[device + 1] - feature_ranges[device]);
    }

    std::vector<real_type> q(num_data_points);
    device_reduction(q_d, q);
    return q;
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::calculate_w(const std::vector<device_ptr_type<real_type>> &data_d,
                                                                    const std::vector<device_ptr_type<real_type>> &data_last_d,
                                                                    const std::vector<device_ptr_type<real_type>> &alpha_d,
                                                                    const std::size_t num_data_points,
                                                                    const std::vector<std::size_t> &feature_ranges) const {
    PLSSVM_ASSERT(!data_d.empty(), "The data_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_d.cbegin(), data_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in data_d must at least contain one data point!");
    PLSSVM_ASSERT(!data_last_d.empty(), "The data_last_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(data_last_d.cbegin(), data_last_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in data_last_d must at least contain one data point!");
    PLSSVM_ASSERT(data_d.size() == data_last_d.size(), "The number of used devices to the data_d and data_last_d vectors must be equal!: {} != {}", data_d.size(), data_last_d.size());
    PLSSVM_ASSERT(!alpha_d.empty(), "The alpha_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(alpha_d.cbegin(), alpha_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in alpha_d must at least contain one data point!");
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
        device_ptr_type<real_type> w_d = device_ptr_type<real_type>{ num_features_in_range, devices_[device] };

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
template <typename real_type>
void gpu_csvm<device_ptr_t, queue_t>::run_device_kernel(const std::size_t device, const parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const std::vector<std::size_t> &feature_ranges, const real_type QA_cost, const real_type add, const std::size_t dept, const std::size_t boundary_size) const {
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
template <typename real_type>
void gpu_csvm<device_ptr_t, queue_t>::device_reduction(std::vector<device_ptr_type<real_type>> &buffer_d, std::vector<real_type> &buffer) const {
    PLSSVM_ASSERT(!buffer_d.empty(), "The buffer_d array may not be empty!");
    PLSSVM_ASSERT(std::all_of(buffer_d.cbegin(), buffer_d.cend(), [](const device_ptr_type<real_type> &ptr) { return !ptr.empty(); }), "Each device_ptr in buffer_d must at least contain one data point!");
    PLSSVM_ASSERT(!buffer.empty(), "The buffer array may not be empty!");

    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].copy_to_host(buffer, 0, buffer.size());

    if (buffer_d.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<device_ptr_type<real_type>>::size_type device = 1; device < buffer_d.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].copy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for default(none) shared(buffer_d, buffer)
        for (typename std::vector<device_ptr_type<real_type>>::size_type device = 0; device < buffer_d.size(); ++device) {
            buffer_d[device].copy_to_device(buffer, 0, buffer.size());
        }
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::pair<detail::aos_matrix<real_type>, std::vector<real_type>> gpu_csvm<device_ptr_t, queue_t>::solve_system_of_linear_equations_impl(const parameter<real_type> &params,
                                                                                                                    const detail::aos_matrix<real_type> &A,
                                                                                                                    detail::aos_matrix<real_type> B_in,
                                                                                                                    const real_type eps,
                                                                                                                    const unsigned long long max_iter) const {
    PLSSVM_ASSERT(!A.empty(), "The data must not be empty!");
    PLSSVM_ASSERT(!B_in.empty(), "At least one right hand side must be given!");
    PLSSVM_ASSERT(eps > real_type{ 0.0 }, "The stopping criterion in the CG algorithm must be greater than 0.0, but is {}!", eps);
    PLSSVM_ASSERT(max_iter > 0, "The number of CG iterations must be greater than 0!");

    using namespace plssvm::operators;

    const std::size_t dept = A.num_rows() - 1;
    constexpr auto boundary_size = 0; // static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    const std::size_t num_features = A.num_cols();
    const std::size_t num_rhs = B_in.num_rows();

    [[maybe_unused]] const std::size_t num_used_devices = 1; // TODO: this->select_num_used_devices(params.kernel_type, num_features);

    std::vector<device_ptr_type<real_type>> data_d;
    std::vector<device_ptr_type<real_type>> data_last_d;
    std::vector<std::size_t> feature_ranges;
    std::tie(data_d, data_last_d, feature_ranges) = this->setup_data_on_device(A, dept, num_features, boundary_size, num_used_devices);

    // create q_red vector
    const std::vector<real_type> q_red = this->generate_q(params, data_d, data_last_d, dept, feature_ranges, boundary_size);

    // calculate QA_costs
    std::vector<real_type> A_back(num_features);
    for (std::size_t i = 0; i < num_features; ++i) {
        A_back[i] = A(dept, i);
    }
    const real_type QA_cost = kernel_function(A_back, A_back, params) + real_type{ 1.0 } / params.cost;

    // update b
    // TODO: better than copy?
    std::vector<real_type> b_back_value(num_rhs);
    detail::aos_matrix<real_type> B{ num_rhs, dept };
    #pragma omp parallel for default(none) shared(B_in, B, b_back_value) firstprivate(dept, num_rhs)
    for (std::size_t row = 0; row < B.num_rows(); ++row) {
        b_back_value[row] = B_in(row, dept);
        for (std::size_t col = 0; col < B.num_cols(); ++col) {
            B(row, col) = B_in(row, col) - b_back_value[row];
        }
    }

    // assemble explicit kernel matrix

    const std::chrono::steady_clock::time_point assembly_start_time = std::chrono::steady_clock::now();

    const device_ptr_type<real_type> explicit_A_d = this->assemble_kernel_matrix(params, data_d.front(), q_red, QA_cost, dept, num_features);
    PLSSVM_ASSERT(dept * dept == explicit_A_d.size(), "Sizes mismatch!: {} != {}", dept, explicit_A_d.size());

    const std::chrono::steady_clock::time_point assembly_end_time = std::chrono::steady_clock::now();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Assembled the kernel matrix in {}.\n",
                detail::tracking_entry{ "cg", "kernel_matrix_assembly", std::chrono::duration_cast<std::chrono::milliseconds>(assembly_end_time - assembly_start_time) });


    // CG

    detail::aos_matrix<real_type> X{ num_rhs, dept, real_type{ 1.0 } };
    device_ptr_type<real_type> X_d{ X.num_entries() };

    detail::aos_matrix<real_type> R{ B };
    device_ptr_type<real_type> R_d{ R.num_entries() };

    // R = B - A * X
    X_d.copy_to_device(X.data());
    R_d.copy_to_device(B.data());
    this->blas_gemm(dept, num_rhs, dept, real_type{ -1.0 }, explicit_A_d, X_d, real_type{ 1.0 }, R_d);
    R_d.copy_to_host(R.data());

    // delta = R.T * R
    std::vector<real_type> delta(num_rhs);
    #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, dept)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp{ 0.0 };
        #pragma omp simd reduction(+ : temp)
        for (std::size_t j = 0; j < dept; ++j) {
            temp += R(i, j) * R(i, j);
        }
        delta[i] = temp;
    }
    const std::vector<real_type> delta0(delta);
    detail::aos_matrix<real_type> Q{ num_rhs, dept };
    device_ptr_type<real_type> Q_d{ Q.num_entries() };

    detail::aos_matrix<real_type> D{ R };
    device_ptr_type<real_type> D_d{ D.num_entries() };

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        const auto iteration_end_time = std::chrono::steady_clock::now();
        const auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Done in {}.\n",
                    iteration_duration);
        average_iteration_time += iteration_duration;
    };
    // get the index of the rhs that has the largest residual difference wrt to its target residual
    const auto residual_info = [&]() {
        real_type max_difference{ 0.0 };
        std::size_t idx{ 0 };
        for (std::size_t i = 0; i < delta.size(); ++i) {
            const real_type difference = delta[i] - (eps * eps * delta0[i]);
            if (difference > max_difference) {
                idx = i;
            }
        }
        return idx;
    };
    // get the number of rhs that have already been converged
    const auto num_converged = [&]() {
        std::vector<bool> converged(delta.size(), false);
        for (std::size_t i = 0; i < delta.size(); ++i) {
            // check if the rhs converged in the current iteration
            converged[i] = delta[i] <= eps * eps * delta0[i];
        }
        return static_cast<std::size_t>(std::count(converged.cbegin(), converged.cend(), true));
    };

    unsigned long long iter = 0;
    for (; iter < max_iter; ++iter) {
        const std::size_t max_residual_difference_idx = residual_info();
        detail::log(verbosity_level::full | verbosity_level::timing,
                    "Start Iteration {} (max: {}) with {}/{} converged rhs (max residual {} with target residual {} for rhs {}). ",
                    iter + 1,
                    max_iter,
                    num_converged(),
                    num_rhs,
                    delta[max_residual_difference_idx],
                    eps * eps * delta0[max_residual_difference_idx],
                    max_residual_difference_idx);
        iteration_start_time = std::chrono::steady_clock::now();

        // Q = A * D
        D_d.copy_to_device(D.data());
        Q_d.memset(0);
        this->blas_gemm(dept, num_rhs, dept, real_type{ 1.0 }, explicit_A_d, D_d, real_type{ 0.0 }, Q_d);
        Q_d.copy_to_host(Q.data());

        // (alpha = delta_new / (D^T * Q))
        std::vector<real_type> alpha(num_rhs);
        #pragma omp parallel for default(none) shared(D, Q, alpha, delta) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t dim = 0; dim < dept; ++dim) {
                temp += D(i, dim) * Q(i, dim);
            }
            alpha[i] = delta[i] / temp;
        }

        // X = X + alpha * D
        #pragma omp parallel for collapse(2) default(none) shared(X, alpha, D) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < dept; ++dim) {
                X(i, dim) += alpha[i] * D(i, dim);
            }
        }

        if (iter % 50 == 49) {
            // R = B - A * X
            X_d.copy_to_device(X.data());
            R_d.copy_to_device(B.data());
            this->blas_gemm(dept, num_rhs, dept, real_type{ -1.0 }, explicit_A_d, X_d, real_type{ 1.0 }, R_d);
            R_d.copy_to_host(R.data());
        } else {
            // R = R - alpha * Q
            #pragma omp parallel for collapse(2) default(none) shared(R, alpha, Q) firstprivate(num_rhs, dept)
            for (std::size_t i = 0; i < num_rhs; ++i) {
                for (std::size_t dim = 0; dim < dept; ++dim) {
                    R(i, dim) -= alpha[i] * Q(i, dim);
                }
            }
        }

        // delta = R^T * R
        const std::vector<real_type> delta_old = delta;
        #pragma omp parallel for default(none) shared(R, delta) firstprivate(num_rhs, dept)
        for (std::size_t col = 0; col < num_rhs; ++col) {
            real_type temp{ 0.0 };
            #pragma omp simd reduction(+ : temp)
            for (std::size_t row = 0; row < dept; ++row) {
                temp += R(col, row) * R(col, row);
            }
            delta[col] = temp;
        }

        // if we are exact enough stop CG iterations, i.e., if every rhs has converged
        if (num_converged() == num_rhs) {
            output_iteration_duration();
            break;
        }

        // (beta = delta_new / delta_old)
        const std::vector<real_type> beta = delta / delta_old;
        // D = beta * D + R
        #pragma omp parallel for collapse(2) default(none) shared(D, beta, R) firstprivate(num_rhs, dept)
        for (std::size_t i = 0; i < num_rhs; ++i) {
            for (std::size_t dim = 0; dim < dept; ++dim) {
                D(i, dim) = beta[i] * D(i, dim) + R(i, dim);
            }
        }

        output_iteration_duration();
    }
    const std::size_t max_residual_difference_idx = residual_info();
    detail::log(verbosity_level::full | verbosity_level::timing,
                "Finished after {}/{} iterations with {}/{} converged rhs (max residual {} with target residual {} for rhs {}) and an average iteration time of {}.\n",
                detail::tracking_entry{ "cg", "iterations", std::min(iter + 1, max_iter) },
                detail::tracking_entry{ "cg", "max_iterations", max_iter },
                detail::tracking_entry{ "cg", "num_converged_rhs", num_converged() },
                detail::tracking_entry{ "cg", "num_rhs", num_rhs },
                delta[max_residual_difference_idx],
                eps * eps * delta0[max_residual_difference_idx],
                max_residual_difference_idx,
                detail::tracking_entry{ "cg", "avg_iteration_time", average_iteration_time / std::min(iter + 1, max_iter) });
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "residuals", delta }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "target_residuals", eps * eps * delta0 }));
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking_entry{ "cg", "epsilon", eps }));
    detail::log(verbosity_level::libsvm,
                "optimization finished, #iter = {}\n",
                std::min(iter + 1, max_iter));

    // calculate bias
    detail::aos_matrix<real_type> X_ret{ num_rhs, A.num_rows() };
    std::vector<real_type> bias(num_rhs);
    #pragma omp parallel for default(none) shared(X, q_red, X_ret, bias, b_back_value) firstprivate(num_rhs, dept, QA_cost)
    for (std::size_t i = 0; i < num_rhs; ++i) {
        real_type temp_sum{ 0.0 };
        real_type temp_dot{ 0.0 };
        #pragma omp simd reduction(+ : temp_sum) reduction(+ : temp_dot)
        for (std::size_t dim = 0; dim < dept; ++dim) {
            temp_sum += X(i, dim);
            temp_dot += q_red[dim] * X(i, dim);

            X_ret(i, dim) = X(i, dim);
        }
        bias[i] = -(b_back_value[i] + QA_cost * temp_sum - temp_dot);
        X_ret(i, dept) = -temp_sum;
    }

    return std::make_pair(std::move(X_ret), std::move(bias));
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
detail::aos_matrix<real_type> gpu_csvm<device_ptr_t, queue_t>::predict_values_impl(const parameter<real_type> &params,
                                                                                   const detail::aos_matrix<real_type> &support_vectors,
                                                                                   const detail::aos_matrix<real_type> &alpha,
                                                                                   const std::vector<real_type> &rho,
                                                                                   detail::aos_matrix<real_type> &w,
                                                                                   const detail::aos_matrix<real_type> &predict_points) const {
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

    device_ptr_type<real_type> sv_d{ num_support_vectors * num_features };
    sv_d.copy_to_device(support_vectors.data());
    device_ptr_type<real_type> predict_points_d{ num_predict_points * num_features };
    predict_points_d.copy_to_device(predict_points.data());

    device_ptr_type<real_type> w_d;  // only used when predicting linear kernel functions
    device_ptr_type<real_type> alpha_d{ num_classes * num_support_vectors };
    alpha_d.copy_to_device(alpha.data());
    device_ptr_type<real_type> rho_d{ num_classes };
    rho_d.copy_to_device(rho);

    if (params.kernel_type == kernel_function_type::linear) {
        // special optimization for the linear kernel function
        if (w.empty()) {
            // fill w vector
            w_d = run_w_kernel(alpha_d, sv_d, num_classes, num_support_vectors, num_features);

            // convert 1D result to aos_matrix out-parameter
            w = detail::aos_matrix<real_type>{ num_classes, num_features };
            w_d.copy_to_host(w.data());
        } else {
            // w already provided -> copy to device
            w_d = device_ptr_type<real_type>{ num_classes * num_features };
            w_d.copy_to_device(w.data());
        }
    }

    // predict
    const device_ptr_type<real_type> out_d = run_predict_kernel(params, w_d, alpha_d, rho_d, sv_d, predict_points_d, num_classes, num_support_vectors, num_predict_points, num_features);;

    // copy results back to host
    detail::aos_matrix<real_type> out_ret{ num_predict_points, num_classes };
    out_d.copy_to_host(out_ret.data());
    return out_ret;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_GPU_CSVM_HPP_
