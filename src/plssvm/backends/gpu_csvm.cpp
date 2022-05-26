#include "plssvm/backends/gpu_csvm.hpp"

#include "plssvm/constants.hpp"               // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range
#include "plssvm/detail/operators.hpp"        // various operator overloads for std::vector and scalars
#include "plssvm/exceptions/exceptions.hpp"   // plssvm::exception
#include "plssvm/parameter.hpp"               // plssvm::parameter

#include "plssvm/detail/layout.hpp"

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    // used for explicitly instantiating the CUDA backend
    #include "plssvm/backends/CUDA/detail/device_ptr.cuh"
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
    // used for explicitly instantiating the HIP backend
    #include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    // used for explicitly instantiating the OpenCL backend
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"
    #include "plssvm/backends/OpenCL/detail/device_ptr.hpp"
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // used for explicitly instantiating the SYCL backend
    #include "sycl/sycl.hpp"
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    #include "plssvm/backends/DPCPP/detail/constants.hpp"
    #include "plssvm/backends/DPCPP/detail/device_ptr.hpp"
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
    #include "plssvm/backends/hipSYCL/detail/constants.hpp"
    #include "plssvm/backends/hipSYCL/detail/device_ptr.hpp"
#endif
#endif

#include "fmt/core.h"    // fmt::print

#include <algorithm>  // std::all_of, std::min, std::max
#include <chrono>     // std::chrono
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

namespace plssvm::detail {

template <typename T, typename device_ptr_t, typename queue_t>
gpu_csvm<T, device_ptr_t, queue_t>::gpu_csvm(parameter<real_type> params) :
    base_type{ std::move(params) } {}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::setup_data_on_device(const std::vector<std::vector<real_type>> &data) {
    const size_type num_data_points = data.size();
    const size_type num_features = data.front().size();

    // set values of member variables
    dept_ = num_data_points - 1;
    boundary_size_ = static_cast<size_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    num_rows_ = dept_ + boundary_size_;
    num_cols_ = num_features;
    feature_ranges_.reserve(devices_.size() + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= devices_.size(); ++device) {
        feature_ranges_.push_back(device * num_cols_ / devices_.size());
    }

    // transform 2D to 1D SoA data
    const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, data, boundary_size_, dept_);

    #pragma omp parallel for default(none) shared(devices_, feature_ranges_, data_last_d_, data_d_, data, transformed_data) firstprivate(dept_, boundary_size_, num_features)
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const size_type num_features_in_range = feature_ranges_[device + 1] - feature_ranges_[device];

        // initialize data_last on device
        data_last_d_[device] = device_ptr_type{ num_features_in_range + boundary_size_, devices_[device] };
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data.back().data() + feature_ranges_[device], 0, num_features_in_range);

        const size_type device_data_size = num_features_in_range * (dept_ + boundary_size_);
        data_d_[device] = device_ptr_type{ device_data_size, devices_[device] };
        data_d_[device].memcpy_to_device(transformed_data.data() + feature_ranges_[device] * (dept_ + boundary_size_), 0, device_data_size);
    }
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::clear_data_from_device() {
    // TODO: ok?
    data_last_d_.clear();
    data_d_.clear();
    devices_.clear();
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::generate_q(const parameter<real_type> &params, [[maybe_unused]] const std::vector<std::vector<real_type>> &data) -> std::vector<real_type> {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<device_ptr_type> q_d(devices_.size());

    #pragma omp parallel for default(none) shared(q_d, devices_, params) firstprivate(dept_, boundary_size_, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);

        // feature splitting on multiple devices
        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_) });

        run_q_kernel(device, range, params, q_d[device], feature_ranges_[device + 1] - feature_ranges_[device]);
    }

    std::vector<real_type> q(dept_);
    device_reduction(q_d, q);
    return q;
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::conjugate_gradient(const parameter<real_type> &params, [[maybe_unused]] const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &b, const std::vector<real_type> &q, real_type QA_cost, real_type eps, size_type max_iter) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<real_type> x(dept_, 1.0);
    std::vector<device_ptr_type> x_d(devices_.size());

    std::vector<real_type> r(dept_, 0.0);
    std::vector<device_ptr_type> r_d(devices_.size());

    #pragma omp parallel for default(none) shared(devices_, x, x_d, r_d) firstprivate(dept_, boundary_size_)
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        x_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept_);

        r_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept_);

    std::vector<device_ptr_type> q_d(devices_.size());
    #pragma omp parallel for default(none) shared(devices_, q, q_d, r_d, x_d, params) firstprivate(dept_, boundary_size_, QA_cost)
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept_);

        // r = Ax (r = b - Ax)
        run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], QA_cost, -1);
    }
    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept_);

    std::vector<device_ptr_type> Ad_d(devices_.size());
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        Ad_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
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

    std::size_t run = 0;
    for (; run < max_iter; ++run) {
        if (verbose) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}). ", run + 1, max_iter, delta, eps * eps * delta0);
        }
        iteration_start_time = std::chrono::steady_clock::now();

        // Ad = A * r (q = A * d)
        #pragma omp parallel for default(none) shared(devices_, Ad_d, r_d, q_d, params) firstprivate(dept_, QA_cost)
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept_);

            run_device_kernel(device, params, q_d[device], Ad_d[device], r_d[device], QA_cost, 1);
        }
        // update Ad (q)
        device_reduction(Ad_d, Ad);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

        #pragma omp parallel for default(none) shared(devices_, x, x_d) firstprivate(dept_)
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            x_d[device].memcpy_to_device(x, 0, dept_);
        }

        if (run % 50 == 49) {
            // r = b
            r_d[0].memcpy_to_device(b, 0, dept_);
            #pragma omp parallel for default(none) shared(devices_, r_d, q_d, x_d, params) firstprivate(QA_cost)
            for (typename std::vector<queue_type>::size_type device = 1; device < devices_.size(); ++device) {
                r_d[device].memset(0);

                // r -= A * x
                run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], QA_cost, -1);
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
        #pragma omp parallel for default(none) shared(devices_, r_d, d) firstprivate(dept_)
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            r_d[device].memcpy_to_device(d, 0, dept_);
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

    return std::vector<real_type>(x.begin(), x.begin() + dept_);
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::calculate_w(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &alpha) -> std::vector<real_type> {
    const size_type num_data_points = A.size();
    const size_type num_features = A.front().size();

    // create w vector and fill with zeros
    std::vector<real_type> w(num_features, real_type{ 0.0 });

    #pragma omp parallel for default(none) shared(devices_, feature_ranges_, alpha, w) firstprivate(num_data_points, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        // feature splitting on multiple devices
        const size_type num_features_in_range = feature_ranges_[device + 1] - feature_ranges_[device];

        // create the w vector on the device
        device_ptr_type w_d = device_ptr_type{ num_features_in_range, devices_[device] };
        // create the weight vector on the device and copy data
        device_ptr_type alpha_d{ num_data_points + THREAD_BLOCK_SIZE, devices_[device] };
        alpha_d.memcpy_to_device(alpha.data(), 0, num_data_points);

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features_in_range) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_features_in_range) });

        // calculate the w vector on the device
        run_w_kernel(device, range, w_d, alpha_d, num_data_points, num_features_in_range);
        device_synchronize(devices_[device]);

        // copy back to host memory
        w_d.memcpy_to_host(w.data() + feature_ranges_[device], 0, num_features_in_range);
    }
    return w;
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, const std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) -> std::vector<real_type> {
    using namespace plssvm::operators;

    const size_type num_support_vectors = support_vectors.size();
    const size_type num_predict_points = predict_points.size();
    const size_type num_features = predict_points.front().size();

    std::vector<real_type> out(predict_points.size());

    if (params.kernel == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        #pragma omp parallel for default(none) shared(out, predict_points, w) firstprivate(num_predict_points, rho)
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < num_predict_points; ++i) {
            out[i] = transposed<real_type>{ w } * predict_points[i] + -rho;
        }
    } else {
        // create result vector on the device
        device_ptr_type out_d{ num_predict_points + boundary_size_, devices_[0] };
        out_d.memset(0);

        // transform prediction data
        const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, predict_points, boundary_size_, predict_points.size());
        device_ptr_type point_d{ num_features * (num_predict_points + boundary_size_), devices_[0] };
        point_d.memset(0);
        point_d.memcpy_to_device(transformed_data, 0, transformed_data.size());

        // create the weight vector on the device and copy data
        device_ptr_type alpha_d{ num_support_vectors + THREAD_BLOCK_SIZE, devices_[0] };
        alpha_d.memset(0);
        alpha_d.memcpy_to_device(alpha, 0, num_support_vectors);

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_support_vectors) / static_cast<real_type>(THREAD_BLOCK_SIZE))),
                                              static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_predict_points) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_support_vectors), std::min<std::size_t>(THREAD_BLOCK_SIZE, num_predict_points) });

        // perform prediction on the first device
        run_predict_kernel(range, params, out_d, alpha_d, point_d, num_support_vectors, num_predict_points, num_features);

        out_d.memcpy_to_host(out, 0, num_predict_points);

        // add bias_ to all predictions
        out += -rho;
    }
    return out;
}


template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::run_device_kernel(const std::size_t device, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type QA_cost, const real_type add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel(device, range, params, q_d, r_d, x_d, QA_cost, add, feature_ranges_[device + 1] - feature_ranges_[device]);
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) {
    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (devices_.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<queue_type>::size_type device = 1; device < devices_.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for default(none) shared(devices_, buffer_d, buffer)
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

// explicitly instantiate template class depending on available backends
#if defined(PLSSVM_HAS_CUDA_BACKEND)
template class gpu_csvm<float, ::plssvm::cuda::detail::device_ptr<float>, int>;
template class gpu_csvm<double, ::plssvm::cuda::detail::device_ptr<double>, int>;
#endif
#if defined(PLSSVM_HAS_HIP_BACKEND)
template class gpu_csvm<float, ::plssvm::hip::detail::device_ptr<float>, int>;
template class gpu_csvm<double, ::plssvm::hip::detail::device_ptr<double>, int>;
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
template class gpu_csvm<float, ::plssvm::opencl::detail::device_ptr<float>, ::plssvm::opencl::detail::command_queue>;
template class gpu_csvm<double, ::plssvm::opencl::detail::device_ptr<double>, ::plssvm::opencl::detail::command_queue>;
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
#if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
template class gpu_csvm<float, ::plssvm::dpcpp::detail::device_ptr<float>, std::unique_ptr<::plssvm::dpcpp::detail::sycl::queue>>;
template class gpu_csvm<double, ::plssvm::dpcpp::detail::device_ptr<double>, std::unique_ptr<::plssvm::dpcpp::detail::sycl::queue>>;
#endif
#if defined(PLSSVM_SYCL_BACKEND_HAS_HIPSYCL)
template class gpu_csvm<float, ::plssvm::hipsycl::detail::device_ptr<float>, std::unique_ptr<::plssvm::hipsycl::detail::sycl::queue>>;
template class gpu_csvm<double, ::plssvm::hipsycl::detail::device_ptr<double>, std::unique_ptr<::plssvm::hipsycl::detail::sycl::queue>>;
#endif
#endif

}  // namespace plssvm::detail
