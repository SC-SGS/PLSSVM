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

#pragma once

#include "plssvm/csvm.hpp"  // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"
#include "plssvm/detail/layout.hpp"
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <vector>  // std::vector

namespace plssvm::detail {

/**
 * @brief A C-SVM implementation for all GPU backends to reduce code duplication.
 * @details Implements all virtual functions defined in plssvm::csvm. The GPU backends only have to implement the actual kernel launches.
 * @tparam T the type of the data
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 */
template <template<typename> typename device_ptr_t, typename queue_t>
class gpu_csvm : public ::plssvm::csvm {
  public:
    /// The type of the device pointer (dependent on the used backend).
    template <typename real_type>
    using device_ptr_type = device_ptr_t<real_type>;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;

    /**
     * @brief Construct a new C-SVM using one of the GPU backends with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit gpu_csvm(plssvm::parameter params = {}) :
        ::plssvm::csvm{ params } {}
    /**
     * @brief Construct a new C-SVM using one of the GPU backend with the optionally provided @p named_args.
     * @param[in] kernel the kernel type used in the C-SVM
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    template <typename... Args, PLSSVM_REQUIRES(detail::has_only_parameter_named_args_v<Args...>)>
    explicit gpu_csvm(Args &&...named_args) :
        ::plssvm::csvm{ std::forward<Args>(named_args)... } {}

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~gpu_csvm() = default;

    /**
     * @brief Return the number of available devices for the current backend.
     * @return the number of available devices (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_available_devices() const noexcept {
        return devices_.size();
    }

  protected:
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    [[nodiscard]] std::pair<std::vector<float>, float> solve_system_of_linear_equations(const parameter<float> &params, const std::vector<std::vector<float>> &A, std::vector<float> b, float eps, unsigned long long max_iter) const final { return this->solve_system_of_linear_equations_impl(params, A, b, eps, max_iter); }
    [[nodiscard]] std::pair<std::vector<double>, double> solve_system_of_linear_equations(const parameter<double> &params, const std::vector<std::vector<double>> &A, std::vector<double> b, double eps, unsigned long long max_iter) const final { return this->solve_system_of_linear_equations_impl(params, A, b, eps, max_iter); }
    template <typename real_type>
    [[nodiscard]] std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, unsigned long long max_iter) const;

    [[nodiscard]] std::vector<float> predict_values(const parameter<float> &params, const std::vector<std::vector<float>> &support_vectors, const std::vector<float> &alpha, float rho, std::vector<float> &w, const std::vector<std::vector<float>> &predict_points) const final { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    [[nodiscard]] std::vector<double> predict_values(const parameter<double> &params, const std::vector<std::vector<double>> &support_vectors, const std::vector<double> &alpha, double rho, std::vector<double> &w, const std::vector<std::vector<double>> &predict_points) const final { return this->predict_values_impl(params, support_vectors, alpha, rho, w, predict_points); }
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const;

    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    template <typename real_type>
    [[nodiscard]] std::tuple<std::vector<device_ptr_type<real_type>>, std::vector<device_ptr_type<real_type>>, std::vector<std::size_t>> setup_data_on_device(const std::vector<std::vector<real_type>> &data, std::size_t num_data_points, std::size_t num_features, std::size_t boundary_size, std::size_t num_used_devices) const;
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> generate_q(const parameter<real_type> &params, const std::vector<device_ptr_type<real_type>> &data_d, const std::vector<device_ptr_type<real_type>> &data_last_d, std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges, std::size_t boundary_size, std::size_t num_used_devices) const;
    /**
     * @copydoc plssvm::csvm::update_w
     */
    template <typename real_type>
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<device_ptr_type<real_type>> &data_d, const std::vector<device_ptr_type<real_type>> &data_last_d, const std::vector<device_ptr_type<real_type>> &alpha_d, std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges, std::size_t num_used_devices) const;

    /**
     * @brief Run the SVM kernel on the GPU denoted by the @p device ID.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
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

    [[nodiscard]] std::size_t select_num_used_devices(kernel_function_type kernel, std::size_t num_features) const noexcept;

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Synchronize the device denoted by @p queue.
     * @param[in,out] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(const queue_type &queue) const = 0;
    /**
     * @brief Run the GPU kernel filling the `q` vector.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[out] q_d the `q` vector to fill
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_q_kernel(std::size_t device, const detail::execution_range &range, const parameter<float> &params, device_ptr_type<float> &q_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    virtual void run_q_kernel(std::size_t device, const detail::execution_range &range, const parameter<double> &params, device_ptr_type<double> &q_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    /**
     * @brief Run the main GPU kernel used in the CG algorithm.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] q_d the `q` vector
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] num_features number of features used for the calculation in the @p device
     */
    virtual void run_svm_kernel(std::size_t device, const detail::execution_range &range, const parameter<float> &params, const device_ptr_type<float> &q_d, device_ptr_type<float> &r_d, const device_ptr_type<float> &x_d, const device_ptr_type<float> &data_d, float QA_cost, float add, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    virtual void run_svm_kernel(std::size_t device, const detail::execution_range &range, const parameter<double> &params, const device_ptr_type<double> &q_d, device_ptr_type<double> &r_d, const device_ptr_type<double> &x_d, const device_ptr_type<double> &data_d, double QA_cost, double add, std::size_t num_data_points_padded, std::size_t num_features) const = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) the calculate the `w` vector used to speed up the prediction when using the linear kernel function.
     * @param[in] device the device ID denoting the GPU on which the kernel should be executed
     * @param[in] range the execution range used to launch the
     * @param[out] w_d the `w` vector to fill, used to speed up the prediction when using the linear kernel
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] num_features number of features used for the calculation on the @p device
     */
    virtual void run_w_kernel(std::size_t device, const detail::execution_range &range, device_ptr_type<float> &w_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_data_points, std::size_t num_features) const = 0;
    virtual void run_w_kernel(std::size_t device, const detail::execution_range &range, device_ptr_type<double> &w_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_data_points, std::size_t num_features) const = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) to predict the new data points @p point_d.
     * @param[in] range the execution range used to launch the kernel
     * @param[out] out_d the calculated prediction
     * @param[in] alpha_d the previously calculated weight for each data point
     * @param[in] point_d the data points to predict
     * @param[in] num_predict_points the number of data points to predict
     */
    virtual void run_predict_kernel(const detail::execution_range &range, const parameter<float> &params, device_ptr_type<float> &out_d, const device_ptr_type<float> &alpha_d, const device_ptr_type<float> &point_d, const device_ptr_type<float> &data_d, const device_ptr_type<float> &data_last_d, std::size_t num_support_vectors, std::size_t num_predict_points, std::size_t num_features) const = 0;
    virtual void run_predict_kernel(const detail::execution_range &range, const parameter<double> &params, device_ptr_type<double> &out_d, const device_ptr_type<double> &alpha_d, const device_ptr_type<double> &point_d, const device_ptr_type<double> &data_d, const device_ptr_type<double> &data_last_d, std::size_t num_support_vectors, std::size_t num_predict_points, std::size_t num_features) const = 0;

    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
};

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::tuple<std::vector<device_ptr_t<real_type>>, std::vector<device_ptr_t<real_type>>, std::vector<std::size_t>>
gpu_csvm<device_ptr_t, queue_t>::setup_data_on_device(const std::vector<std::vector<real_type>> &data,
                                                      const std::size_t num_data_points,
                                                      const std::size_t num_features,
                                                      const std::size_t boundary_size,
                                                      const std::size_t num_used_devices) const {
    // calculate the number of features per device
    std::vector<std::size_t> feature_ranges(num_used_devices + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= num_used_devices; ++device) {
        feature_ranges[device] = device * num_features / num_used_devices;
    }

    // transform 2D to 1D SoA data
    const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, data, boundary_size, num_data_points);

    std::vector<device_ptr_type<real_type>> data_last_d(num_used_devices);
    std::vector<device_ptr_type<real_type>> data_d(num_used_devices);

#pragma omp parallel for default(none) shared(num_used_devices, devices_, feature_ranges, data_last_d, data_d, data, transformed_data) firstprivate(num_data_points, boundary_size, num_features)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        const std::size_t num_features_in_range = feature_ranges[device + 1] - feature_ranges[device];

        // initialize data_last on device
        data_last_d[device] = device_ptr_type<real_type>{ num_features_in_range + boundary_size, devices_[device] };
        data_last_d[device].memset(0);
        data_last_d[device].memcpy_to_device(data.back().data() + feature_ranges[device], 0, num_features_in_range);

        const std::size_t device_data_size = num_features_in_range * (num_data_points + boundary_size);
        data_d[device] = device_ptr_type<real_type>{ device_data_size, devices_[device] };
        data_d[device].memcpy_to_device(transformed_data.data() + feature_ranges[device] * (num_data_points + boundary_size), 0, device_data_size);
    }

    return std::make_tuple(std::move(data_d), std::move(data_last_d), std::move(feature_ranges));
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::generate_q(const parameter<real_type> &params,
                                                                   const std::vector<device_ptr_type<real_type>> &data_d,
                                                                   const std::vector<device_ptr_type<real_type>> &data_last_d,
                                                                   const std::size_t num_data_points,
                                                                   const std::vector<std::size_t> &feature_ranges,
                                                                   const std::size_t boundary_size,
                                                                   const std::size_t num_used_devices) const {
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
std::pair<std::vector<real_type>, real_type> gpu_csvm<device_ptr_t, queue_t>::solve_system_of_linear_equations_impl(const parameter<real_type> &params,
                                                                                                                    const std::vector<std::vector<real_type>> &A,
                                                                                                                    std::vector<real_type> b,
                                                                                                                    const real_type eps,
                                                                                                                    const unsigned long long max_iter) const {
    using namespace plssvm::operators;

    const std::size_t dept = A.size() - 1;
    constexpr std::size_t boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    const std::size_t num_features = A.front().size();

    const std::size_t num_used_devices = select_num_used_devices(params.kernel_type, num_features);

    std::vector<device_ptr_type<real_type>> data_d;
    std::vector<device_ptr_type<real_type>> data_last_d;
    std::vector<std::size_t> feature_ranges;
    std::tie(data_d, data_last_d, feature_ranges) = this->setup_data_on_device(A, dept, num_features, boundary_size, num_used_devices);

    // create q vector
    const std::vector<real_type> q = this->generate_q(params, data_d, data_last_d, dept, feature_ranges, boundary_size, num_used_devices);

    // calculate QA_costs
    const real_type QA_cost = kernel_function(A.back(), A.back(), params) + real_type{ 1.0 } / params.cost;

    // update b
    const real_type b_back_value = b.back();
    b.pop_back();
    b -= b_back_value;

    std::vector<real_type> x(dept, 1.0);
    std::vector<device_ptr_type<real_type>> x_d(num_used_devices);

    std::vector<real_type> r(dept, 0.0);
    std::vector<device_ptr_type<real_type>> r_d(num_used_devices);

#pragma omp parallel for default(none) shared(num_used_devices, devices_, x, x_d, r_d) firstprivate(dept, boundary_size)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        x_d[device] = device_ptr_type<real_type>{ dept + boundary_size, devices_[device] };
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept);

        r_d[device] = device_ptr_type<real_type>{ dept + boundary_size, devices_[device] };
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept);

    std::vector<device_ptr_type<real_type>> q_d(num_used_devices);
#pragma omp parallel for default(none) shared(num_used_devices, devices_, q, q_d, r_d, x_d, data_d, feature_ranges, params) firstprivate(dept, boundary_size, QA_cost, num_features)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        q_d[device] = device_ptr_type<real_type>{ dept + boundary_size, devices_[device] };
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept);

        // r = Ax (r = b - Ax)
        run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, real_type{ -1.0 }, dept, boundary_size);
    }
    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<device_ptr_type<real_type>> Ad_d(num_used_devices);
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        Ad_d[device] = device_ptr_type<real_type>{ dept + boundary_size, devices_[device] };
    }

    std::vector<real_type> d(r);

    // timing for each CG iteration
    std::chrono::milliseconds average_iteration_time{};
    std::chrono::steady_clock::time_point iteration_start_time{};
    const auto output_iteration_duration = [&]() {
        auto iteration_end_time = std::chrono::steady_clock::now();
        auto iteration_duration = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end_time - iteration_start_time);
        fmt::print("Done in {}.\n", iteration_duration);
        average_iteration_time += iteration_duration;
    };

    unsigned long long run = 0;
    for (; run < max_iter; ++run) {
        if (verbose) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", run + 1, max_iter, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

// Ad = A * r (q = A * d)
#pragma omp parallel for default(none) shared(num_used_devices, devices_, Ad_d, r_d, q_d, data_d, feature_ranges, params) firstprivate(dept, QA_cost, boundary_size, num_features)
        for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept);

            run_device_kernel(device, params, q_d[device], Ad_d[device], r_d[device], data_d[device], feature_ranges, QA_cost, real_type{ 1.0 }, dept, boundary_size);
        }
        // update Ad (q)
        device_reduction(Ad_d, Ad);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

#pragma omp parallel for default(none) shared(num_used_devices, devices_, x, x_d) firstprivate(dept)
        for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
            x_d[device].memcpy_to_device(x, 0, dept);
        }

        if (run % 50 == 49) {
#pragma omp parallel for default(none) shared(devices_, r_d, b, q_d, x_d, params, data_d, feature_ranges) firstprivate(QA_cost, dept)
            for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
                if (device == 0) {
                    // r = b
                    r_d[device].memcpy_to_device(b, 0, dept);
                } else {
                    // set r to 0
                    r_d[device].memset(0);
                }
                // r -= A * x
                run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, real_type{ -1.0 }, dept, boundary_size);
            }

            device_reduction(r_d, r);
        } else {
            // r -= alpha_cd * Ad (r = r - alpha * q)
            r -= alpha_cd * Ad;
        }

        // (delta = r^T * r)
        const real_type delta_old = delta;
        delta = transposed{ r } * r;
        // if we are exact enough stop CG iterations
        if (delta <= eps * eps * delta0) {
            if (verbose) {
                output_iteration_duration();
            }
            break;
        }

        // (beta = delta_new / delta_old)
        const real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

// r_d = d
#pragma omp parallel for default(none) shared(num_used_devices, devices_, r_d, d) firstprivate(dept)
        for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
            r_d[device].memcpy_to_device(d, 0, dept);
        }

        if (verbose) {
            output_iteration_duration();
        }
    }
    if (verbose) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}) and an average iteration time of {}.\n",
                   std::min(run + 1, max_iter),
                   delta,
                   eps * eps * delta0,
                   average_iteration_time / std::min(run + 1, max_iter));
    }

    // calculate bias
    std::vector<real_type> alpha(x.begin(), x.begin() + dept);
    const real_type bias = b_back_value + QA_cost * sum(alpha) - (transposed{ q } * alpha);
    alpha.push_back(-sum(alpha));

    return std::make_pair(std::move(alpha), -bias);
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::calculate_w(const std::vector<device_ptr_type<real_type>> &data_d, const std::vector<device_ptr_type<real_type>> &data_last_d, const std::vector<device_ptr_type<real_type>> &alpha_d, const std::size_t num_data_points, const std::vector<std::size_t> &feature_ranges, const std::size_t num_used_devices) const {
    // create w vector and fill with zeros
    std::vector<real_type> w(std::accumulate(feature_ranges.begin(), feature_ranges.end(), std::size_t{ 0 }), real_type{ 0.0 });

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
        w_d.memcpy_to_host(w.data() + feature_ranges[device], 0, num_features_in_range);
    }
    return w;
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
std::vector<real_type> gpu_csvm<device_ptr_t, queue_t>::predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const {
    using namespace plssvm::operators;

    const std::size_t num_support_vectors = support_vectors.size();
    const std::size_t num_predict_points = predict_points.size();
    const std::size_t num_features = predict_points.front().size();
    const std::size_t boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;

    const std::size_t num_used_devices = select_num_used_devices(params.kernel_type, num_features);

    auto [data_d, data_last_d, feature_ranges] = this->setup_data_on_device(support_vectors, num_support_vectors - 1, num_features, boundary_size, num_used_devices);

    std::vector<device_ptr_type<real_type>> alpha_d(num_used_devices);
#pragma omp parallel for default(none) shared(num_used_devices, devices_, alpha_d, alpha) firstprivate(num_support_vectors)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        alpha_d[device] = device_ptr_type<real_type>{ num_support_vectors + THREAD_BLOCK_SIZE, devices_[device] };
        alpha_d[device].memset(0);
        alpha_d[device].memcpy_to_device(alpha, 0, num_support_vectors);
    }

    std::vector<real_type> out(predict_points.size());

    // use faster methode in case of the linear kernel function
    if (params.kernel_type == kernel_function_type::linear && w.empty()) {
        w = calculate_w(data_d, data_last_d, alpha_d, support_vectors.size(), feature_ranges, num_used_devices);
    }

    if (params.kernel_type == kernel_function_type::linear) {
// use faster methode in case of the linear kernel function
#pragma omp parallel for default(none) shared(out, predict_points, w) firstprivate(num_predict_points, rho)
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < num_predict_points; ++i) {
            out[i] = transposed<real_type>{ w } * predict_points[i] + -rho;
        }
    } else {
        // create result vector on the device
        device_ptr_type<real_type> out_d{ num_predict_points + boundary_size, devices_[0] };
        out_d.memset(0);

        // transform prediction data
        const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, predict_points, boundary_size, predict_points.size());
        device_ptr_type<real_type> point_d{ num_features * (num_predict_points + boundary_size), devices_[0] };
        point_d.memset(0);
        point_d.memcpy_to_device(transformed_data, 0, transformed_data.size());

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_support_vectors) / static_cast<real_type>(THREAD_BLOCK_SIZE))),
                                              static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_predict_points) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_support_vectors), std::min<std::size_t>(THREAD_BLOCK_SIZE, num_predict_points) });

        // perform prediction on the first device
        run_predict_kernel(range, params, out_d, alpha_d[0], point_d, data_d[0], data_last_d[0], num_support_vectors, num_predict_points, num_features);

        out_d.memcpy_to_host(out, 0, num_predict_points);

        // add bias_ to all predictions
        out += -rho;
    }
    return out;
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
void gpu_csvm<device_ptr_t, queue_t>::run_device_kernel(const std::size_t device, const parameter<real_type> &params, const device_ptr_type<real_type> &q_d, device_ptr_type<real_type> &r_d, const device_ptr_type<real_type> &x_d, const device_ptr_type<real_type> &data_d, const std::vector<std::size_t> &feature_ranges, const real_type QA_cost, const real_type add, const std::size_t dept, const std::size_t boundary_size) const {
    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept) / static_cast<real_type>(boundary_size)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel(device, range, params, q_d, r_d, x_d, data_d, QA_cost, add, dept + boundary_size, feature_ranges[device + 1] - feature_ranges[device]);
}

template <template <typename> typename device_ptr_t, typename queue_t>
template <typename real_type>
void gpu_csvm<device_ptr_t, queue_t>::device_reduction(std::vector<device_ptr_type<real_type>> &buffer_d, std::vector<real_type> &buffer) const {
    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (buffer_d.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<device_ptr_type<real_type>>::size_type device = 1; device < buffer_d.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

#pragma omp parallel for default(none) shared(buffer_d, buffer)
        for (typename std::vector<device_ptr_type<real_type>>::size_type device = 0; device < buffer_d.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

template <template <typename> typename device_ptr_t, typename queue_t>
std::size_t gpu_csvm<device_ptr_t, queue_t>::select_num_used_devices(const kernel_function_type kernel, const std::size_t num_features) const noexcept {
    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel == kernel_function_type::polynomial || kernel == kernel_function_type::rbf) {
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

}  // namespace plssvm::detail