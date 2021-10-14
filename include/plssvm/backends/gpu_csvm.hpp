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

#include "plssvm/constants.hpp"               // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range
#include "plssvm/detail/operators.hpp"        // various operator overloads for std::vector and scalars
#include "plssvm/parameter.hpp"               // plssvm::parameter

#include <algorithm>  // std::all_of, std::min
#include <cmath>      // std::ceil
#include <vector>     // std::vector

namespace plssvm::detail {

/**
 * @brief A C-SVM class for all GPU backends to reduce code duplication. Implements all virtual functions defined in `plssvm::csvm`.
 * @tparam T the type of the data
 * @tparam device_ptr_t the type of the device pointer (dependent on the used backend)
 * @tparam queue_t the type of the device queue (dependent on the used backend)
 */
template <typename T, typename device_ptr_t, typename queue_t>
class gpu_csvm : public csvm<T> {
  protected:
    /// The template base type of the C-SVM class.
    using base_type = ::plssvm::csvm<T>;

    using base_type::alpha_ptr_;
    using base_type::bias_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_ptr_;
    using base_type::degree_;
    using base_type::epsilon_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;
    using base_type::value_ptr_;
    using base_type::w_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /// The type of the device pointer (dependent on the used backend).
    using device_ptr_type = device_ptr_t;
    /// The type of the device queue (dependent on the used backend).
    using queue_type = queue_t;

    /**
     * @brief Construct a new C-SVM using any GPU backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit gpu_csvm(const parameter<T> &params) :
        base_type{ params } {}

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~gpu_csvm() = default;

    //*************************************************************************************************************************************//
    //                                                functions inherited from plssvm::csvm                                                //
    //*************************************************************************************************************************************//
    /**
     * @copydoc plssvm::csvm::predict(const std::vector<std::vector<real_type>>&)
     */
    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) final {
        using namespace plssvm::operators;

        PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");                                                                 // exception in constructor
        PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");                                                                    // exception in constructor
        PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());  // exception in constructor

        if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
            throw exception{ "All points in the prediction point vector must have the same number of features!" };
        } else if (alpha_ptr_ == nullptr) {
            throw exception{ "No alphas provided for prediction!" };
        }

        // return empty vector if there are no points to predict
        if (points.empty()) {
            return {};
        }
        if (points.front().size() != data_ptr_->front().size()) {
            throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
        }

        // check if data already resides on the first device
        if (data_d_[0].empty()) {
            setup_data_on_device();
        }

        std::vector<real_type> out(points.size());

        if (kernel_ == kernel_type::linear) {
            // use faster methode in case of the linear kernel function
            if (w_.empty()) {
                update_w();
            }
            #pragma omp parallel for
            for (size_type i = 0; i < points.size(); ++i) {
                out[i] = transposed<real_type>{ w_ } * points[i] + bias_;
            }
        } else {
            // create result vector on the device
            device_ptr_type out_d{ points.size() + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE, devices_[0] };
            out_d.memset(0);

            // transform prediction data
            const std::vector<real_type> transformed_data = base_type::transform_data(points, THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE, points.size());
            device_ptr_type point_d{ points[0].size() * (points.size() + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE), devices_[0] };
            point_d.memcpy_to_device(transformed_data, 0, transformed_data.size());

            // create the weight vector on the device and copy data
            device_ptr_type alpha_d{ num_data_points_ + THREAD_BLOCK_SIZE, devices_[0] };
            alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

            const detail::execution_range range({ static_cast<size_type>(std::ceil(static_cast<real_type>(num_data_points_) / static_cast<real_type>(THREAD_BLOCK_SIZE))), static_cast<size_type>(std::ceil(static_cast<real_type>(points.size()) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                                { std::min<size_type>(THREAD_BLOCK_SIZE, num_data_points_), std::min<size_type>(THREAD_BLOCK_SIZE, points.size()) });

            // perform prediction on the first device
            run_predict_kernel(range, out_d, alpha_d, point_d, points.size());

            out_d.memcpy_to_host(out, 0, points.size());

            // add bias_ to all predictions
            #pragma omp parallel for
            for (size_type i = 0; i < points.size(); ++i) {
                out[i] += bias_;
            }
        }

        return out;
    }

  protected:
    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    void setup_data_on_device() final {
        // set values of member variables
        // TODO: signed vs unsigned (also in other backends)
        dept_ = num_data_points_ - 1;
        boundary_size_ = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
        num_rows_ = static_cast<int>(dept_ + boundary_size_);
        num_cols_ = static_cast<int>(num_features_);

        // transform 2D to 1D data
        const std::vector<real_type> transformed_data = base_type::transform_data(*data_ptr_, boundary_size_, dept_);

        #pragma omp parallel for
        for (size_type device = 0; device < devices_.size(); ++device) {
            const int first_feature = static_cast<int>(device * static_cast<size_type>(num_cols_) / devices_.size());
            const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_cols_) / devices_.size());

            // initialize data_last on device
            data_last_d_[device] = device_ptr_type{ last_feature - first_feature + boundary_size_, devices_[device] };
            data_last_d_[device].memset(0);
            data_last_d_[device].memcpy_to_device(data_ptr_->back().data() + first_feature, 0, last_feature - first_feature);

            std::size_t device_data_size = (last_feature - first_feature) * (dept_ + boundary_size_);
            data_d_[device] = device_ptr_type{ device_data_size, devices_[device] };
            data_d_[device].memcpy_to_device(transformed_data.data() + first_feature * (dept_ + boundary_size_), 0, device_data_size);
        }
    }
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    [[nodiscard]] std::vector<real_type> generate_q() final {
        PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

        std::vector<device_ptr_type> q_d(devices_.size());
        for (size_type device = 0; device < devices_.size(); ++device) {
            q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
            q_d[device].memset(0);
        }

        for (size_type device = 0; device < devices_.size(); ++device) {
            // feature splitting on multiple devices
            const int first_feature = static_cast<int>(device * static_cast<size_type>(num_cols_) / devices_.size());
            const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_cols_) / devices_.size());
            const detail::execution_range range({ static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                                { std::min<size_type>(THREAD_BLOCK_SIZE, dept_) });

            run_q_kernel(device, range, q_d[device], last_feature - first_feature);  // TODO: range as member?
        }

        std::vector<real_type> q(dept_);
        device_reduction(q_d, q);
        return q;
    }
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) final {
        using namespace plssvm::operators;

        PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

        std::vector<real_type> x(dept_, 1.0);
        std::vector<device_ptr_type> x_d(devices_.size());

        std::vector<real_type> r(dept_, 0.0);
        std::vector<device_ptr_type> r_d(devices_.size());

        for (size_type device = 0; device < devices_.size(); ++device) {
            x_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
            r_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        }
        #pragma omp parallel for
        for (size_type device = 0; device < devices_.size(); ++device) {
            x_d[device].memset(0);
            x_d[device].memcpy_to_device(x, 0, dept_);
            r_d[device].memset(0);
        }
        r_d[0].memcpy_to_device(b, 0, dept_);

        std::vector<device_ptr_type> q_d(devices_.size());
        for (size_type device = 0; device < devices_.size(); ++device) {
            q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        }
        #pragma omp parallel for
        for (size_type device = 0; device < devices_.size(); ++device) {
            q_d[device].memset(0);
            q_d[device].memcpy_to_device(q, 0, dept_);
        }

        // r = Ax (r = b - Ax)
        #pragma omp parallel for
        for (size_type device = 0; device < devices_.size(); ++device) {
            run_device_kernel(device, q_d[device], r_d[device], x_d[device], -1);
        }

        device_reduction(r_d, r);

        // delta = r.T * r
        real_type delta = transposed{ r } * r;
        const real_type delta0 = delta;
        std::vector<real_type> Ad(dept_);

        std::vector<device_ptr_type> Ad_d(devices_.size());
        for (size_type device = 0; device < devices_.size(); ++device) {
            Ad_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        }

        std::vector<real_type> d(r);

        size_type run = 0;
        for (; run < imax; ++run) {
            if (print_info_) {
                fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
            }
            // Ad = A * r (q = A * d)
            #pragma omp parallel for
            for (size_type device = 0; device < devices_.size(); ++device) {
                Ad_d[device].memset(0);
                r_d[device].memset(0, dept_);
            }
            #pragma omp parallel for
            for (size_type device = 0; device < devices_.size(); ++device) {
                run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], 1);
            }

            // update Ad (q)
            device_reduction(Ad_d, Ad);

            // (alpha = delta_new / (d^T * q))
            const real_type alpha_cd = delta / (transposed{ d } * Ad);

            // (x = x + alpha * d)
            x += alpha_cd * d;

            #pragma omp parallel for
            for (size_type device = 0; device < devices_.size(); ++device) {
                x_d[device].memcpy_to_device(x, 0, dept_);
            }

            if (run % 50 == 49) {
                // r = b
                r_d[0].memcpy_to_device(b, 0, dept_);
                #pragma omp parallel for
                for (size_type device = 1; device < devices_.size(); ++device) {
                    r_d[device].memset(0);
                }

                // r -= A * x
                #pragma omp parallel for
                for (size_type device = 0; device < devices_.size(); ++device) {
                    run_device_kernel(device, q_d[device], r_d[device], x_d[device], -1);
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
                break;
            }

            // (beta = delta_new / delta_old)
            real_type beta = delta / delta_old;
            // d = beta * d + r
            d = beta * d + r;

            // r_d = d
            #pragma omp parallel for
            for (size_type device = 0; device < devices_.size(); ++device) {
                r_d[device].memcpy_to_device(d, 0, dept_);
            }
        }
        if (print_info_) {
            fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
        }

        return std::vector<real_type>(x.begin(), x.begin() + dept_);
    }

    /**
     * @copydoc plssvm::csvm::update_w
     */
    void update_w() final {
        w_.resize(num_features_);
        #pragma omp parallel for
        for (size_type device = 0; device < devices_.size(); ++device) {
            // feature splitting on multiple devices
            const int first_feature = static_cast<int>(device * static_cast<size_type>(num_features_) / devices_.size());
            const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_features_) / devices_.size());

            // create the w vector on the device
            w_d_ = device_ptr_type{ static_cast<size_type>(last_feature - first_feature), devices_[device] };
            // create the weight vector on the device and copy data
            device_ptr_type alpha_d{ num_data_points_ + THREAD_BLOCK_SIZE, devices_[device] };
            alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

            const detail::execution_range range({ static_cast<size_type>(std::ceil(static_cast<real_type>(last_feature - first_feature) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                                { std::min<size_type>(THREAD_BLOCK_SIZE, last_feature - first_feature) });

            // calculate the w vector on the device
            run_w_kernel(device, range, alpha_d, last_feature - first_feature);
            device_synchronize(devices_[device]);

            // copy back to host memory
            w_d_.memcpy_to_host(w_.data() + first_feature, 0, last_feature - first_feature);
        }
    }

  protected:
    /**
     * @brief Run the SVM kernel the @p device.
     * @param[in] device the OpenCL device to run the kernel on
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[in,out] r_d the result vector
     * @param[in] x_d the `x` vector
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(const size_type device, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add) {
        PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
        PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

        // feature splitting on multiple devices
        const int first_feature = static_cast<int>(device * static_cast<size_type>(num_cols_) / devices_.size());
        const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_cols_) / devices_.size());

        const auto grid = static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
        const auto block = std::min<size_type>(THREAD_BLOCK_SIZE, dept_);
        const detail::execution_range range({ grid, grid }, { block, block });

        run_svm_kernel(device, range, q_d, r_d, x_d, add, first_feature, last_feature);
    }
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each devices.
     * @param[in,out] buffer_d the data to gather
     * @param[in,out] buffer the reduces data
     */
    void device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) {
        device_synchronize(devices_[0]);
        buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

        if (devices_.size() > 1) {
            std::vector<real_type> ret(buffer.size());
            for (size_type device = 1; device < devices_.size(); ++device) {
                device_synchronize(devices_[device]);
                buffer_d[device].memcpy_to_host(ret, 0, ret.size());

                #pragma omp parallel for
                for (size_type j = 0; j < ret.size(); ++j) {
                    buffer[j] += ret[j];
                }
            }

            #pragma omp parallel for
            for (size_type device = 0; device < devices_.size(); ++device) {
                buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
            }
        }
    }

    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Synchronize the device denoted by @p queue.
     * @param[in,out] queue the queue denoting the device to synchronize
     */
    virtual void device_synchronize(queue_type &queue) = 0;
    /**
     * @brief Run the GPU kernel filling the `q_` vector.
     * @param[in] device the device on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[out] q_d the `q` vector to fill
     * @param[in] coll_range TODO:
     */
    virtual void run_q_kernel(const size_type device, const detail::execution_range<size_type> &range, device_ptr_type &q_d, const int coll_range) = 0;
    /**
     * @brief Run the main GPU kernel used in the CG algorithm.
     * @param[in] device the device on which the kernel should be executed
     * @param[in] range the execution range used to launch the kernel
     * @param[in] q_d the `q` vector
     * @param[in,out] r_d the result vector
     * @param[in] x_d the right-hand side of the equation
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     * @param[in] first_feature the first feature used in the calculations (depending on @p device)
     * @param[in] last_feature the last feature used in the calculations (depending on @p device)
     */
    virtual void run_svm_kernel(const size_type device, const detail::execution_range<size_type> &range, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add, const int first_feature, const int last_feature) = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) the calculate the `w` vector used to speedup the prediction when using the linear kernel function.
     * @param[in] range the execution range used to launch the kernel
     * @param[out] alpha_d the previously calculated weight for each data point
     */
    virtual void run_w_kernel(const size_type device, const detail::execution_range<size_type> &range, const device_ptr_type &alpha_d, const size_type num_features) = 0;
    /**
     * @brief Run the GPU kernel (only on the first GPU) to predict the new data points @p point_d.
     * @param[in] range the execution range used to launch the kernel
     * @param[out] out_d the calculated prediction
     * @param[in] alpha_d the previsouly calculated weight for each data point
     * @param[in] point_d the data points to predict
     * @param[in] num_predict_points the number of data points to predict
     */
    virtual void run_predict_kernel(const detail::execution_range<size_type> &range, device_ptr_type &out_d, const device_ptr_type &alpha_d, const device_ptr_type &point_d, const size_type num_predict_points) = 0;

    //*************************************************************************************************************************************//
    //                                             internal variables specific to GPU backends                                             //
    //*************************************************************************************************************************************//
    /// The number of data points excluding the last data point.
    size_type dept_{};
    /// The boundary size used to remove boundary condition checks inside the kernels.
    size_type boundary_size_{};
    /// The number of rows to calculate including the boundary values.v
    int num_rows_{};
    /// The number of columns in the data matrix (= the number of features per data point).
    int num_cols_{};

    /// The available/used backend devices.
    std::vector<queue_type> devices_{};
    /// The data saved across all devices.
    std::vector<device_ptr_type> data_d_{};
    /// The last row of the data matrix.
    std::vector<device_ptr_type> data_last_d_{};
    /// The normal vector used for speeding up the prediction in case of the linear kernel function saved on the first device.
    device_ptr_type w_d_{};
};

}  // namespace plssvm::detail