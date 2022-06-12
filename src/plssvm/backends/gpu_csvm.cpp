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

#include "fmt/core.h"    // fmt::print, fmt::format

#include <algorithm>  // std::all_of, std::min, std::max
#include <chrono>     // std::chrono
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <iostream>   // std::clog, std::endl
#include <vector>     // std::vector

namespace plssvm::detail {

template <typename T, typename device_ptr_t, typename queue_t>
gpu_csvm<T, device_ptr_t, queue_t>::gpu_csvm(parameter<real_type> params) :
    base_type{ std::move(params) } {}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::setup_data_on_device(const std::vector<std::vector<real_type>> &data, const size_type num_data_points, const size_type num_features, const size_type boundary_size, const size_type num_used_devices) const -> std::tuple<std::vector<device_ptr_type>, std::vector<device_ptr_type>, std::vector<size_type>> {
    // calculate the number of features per device
    std::vector<size_type> feature_ranges(num_used_devices + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= num_used_devices; ++device) {
        feature_ranges.push_back(device * num_features / num_used_devices);
    }

    // transform 2D to 1D SoA data
    const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, data, boundary_size, num_data_points);

    std::vector<device_ptr_type> data_last_d(num_used_devices);
    std::vector<device_ptr_type> data_d(num_used_devices);

    #pragma omp parallel for default(none) shared(num_used_devices, devices_, feature_ranges, data_last_d, data_d, data, transformed_data) firstprivate(num_data_points, boundary_size, num_features)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        const size_type num_features_in_range = feature_ranges[device + 1] - feature_ranges[device];

        // initialize data_last on device
        data_last_d[device] = device_ptr_type{ num_features_in_range + boundary_size, devices_[device] };
        data_last_d[device].memset(0);
        data_last_d[device].memcpy_to_device(data.back().data() + feature_ranges[device], 0, num_features_in_range);

        const size_type device_data_size = num_features_in_range * (num_data_points + boundary_size);
        data_d[device] = device_ptr_type{ device_data_size, devices_[device] };
        data_d[device].memcpy_to_device(transformed_data.data() + feature_ranges[device] * (num_data_points + boundary_size), 0, device_data_size);
    }

    return std::make_tuple(std::move(data_d), std::move(data_last_d), std::move(feature_ranges));
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::generate_q(const parameter<real_type> &params, const std::vector<device_ptr_type> &data_d, const std::vector<device_ptr_type> &data_last_d, const size_type num_data_points, const std::vector<size_type> &feature_ranges, const size_type boundary_size, const size_type num_used_devices) const -> std::vector<real_type> {
    std::vector<device_ptr_type> q_d(num_used_devices);

    #pragma omp parallel for default(none) shared(num_used_devices, q_d, devices_, data_d, data_last_d, feature_ranges, params) firstprivate(num_data_points, boundary_size, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        q_d[device] = device_ptr_type{ num_data_points + boundary_size, devices_[device] };
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

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::solve_system_of_linear_equations(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, std::vector<real_type> b, real_type eps, size_type max_iter) const -> std::pair<std::vector<real_type>, real_type> {
    using namespace plssvm::operators;

    const size_type dept = A.size() - 1;
    const size_type boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    const size_type num_features = A.front().size();

    const size_type num_used_devices = select_num_used_devices(params.kernel, num_features);

    std::vector<device_ptr_type> data_d;
    std::vector<device_ptr_type> data_last_d;
    std::vector<size_type> feature_ranges;
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
    std::vector<device_ptr_type> x_d(num_used_devices);

    std::vector<real_type> r(dept, 0.0);
    std::vector<device_ptr_type> r_d(num_used_devices);

    #pragma omp parallel for default(none) shared(num_used_devices, devices_, x, x_d, r_d) firstprivate(dept, boundary_size)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        x_d[device] = device_ptr_type{ dept + boundary_size, devices_[device] };
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept);

        r_d[device] = device_ptr_type{ dept + boundary_size, devices_[device] };
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept);

    std::vector<device_ptr_type> q_d(num_used_devices);
    #pragma omp parallel for default(none) shared(num_used_devices, devices_, q, q_d, r_d, x_d, data_d, feature_ranges, params) firstprivate(dept, boundary_size, QA_cost, num_features)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        q_d[device] = device_ptr_type{ dept + boundary_size, devices_[device] };
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept);

        // r = Ax (r = b - Ax)
        run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, -1, dept, boundary_size);
    }
    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<device_ptr_type> Ad_d(num_used_devices);
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        Ad_d[device] = device_ptr_type{ dept + boundary_size, devices_[device] };
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
        #pragma omp parallel for default(none) shared(num_used_devices, devices_, Ad_d, r_d, q_d, data_d, feature_ranges, params) firstprivate(dept, QA_cost, boundary_size, num_features)
        for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept);

            run_device_kernel(device, params, q_d[device], Ad_d[device], r_d[device], data_d[device], feature_ranges, QA_cost, 1, dept, boundary_size);
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
            // r = b
            r_d[0].memcpy_to_device(b, 0, dept);
            #pragma omp parallel for default(none) shared(num_used_devices, devices_, r_d, q_d, x_d, data_d, feature_ranges, params) firstprivate(QA_cost, dept, boundary_size, num_features)
            for (typename std::vector<queue_type>::size_type device = 1; device < num_used_devices; ++device) {
                r_d[device].memset(0);

                // r -= A * x
                run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, -1, dept, boundary_size);
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

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::calculate_w(const std::vector<device_ptr_type> &data_d, const std::vector<device_ptr_type> &data_last_d, const std::vector<device_ptr_type> &alpha_d, const size_type num_data_points, const std::vector<size_type> &feature_ranges, const size_type num_used_devices) const-> std::vector<real_type> {
    // create w vector and fill with zeros
    std::vector<real_type> w(std::accumulate(feature_ranges.begin(), feature_ranges.end(), size_type{ 0 }), real_type{ 0.0 });

    #pragma omp parallel for default(none) shared(num_used_devices, devices_, feature_ranges, alpha_d, data_d, data_last_d, w) firstprivate(num_data_points, THREAD_BLOCK_SIZE)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        // feature splitting on multiple devices
        const size_type num_features_in_range = feature_ranges[device + 1] - feature_ranges[device];

        // create the w vector on the device
        device_ptr_type w_d = device_ptr_type{ num_features_in_range, devices_[device] };

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

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::vector<real_type> &w, const std::vector<std::vector<real_type>> &predict_points) const -> std::vector<real_type> {
    using namespace plssvm::operators;

    const size_type num_support_vectors = support_vectors.size();
    const size_type num_predict_points = predict_points.size();
    const size_type num_features = predict_points.front().size();
    const size_type boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;

    const size_type num_used_devices = select_num_used_devices(params.kernel, num_features);

    auto [data_d, data_last_d, feature_ranges] = this->setup_data_on_device(support_vectors, num_support_vectors - 1, num_features, boundary_size, num_used_devices);

    std::vector<device_ptr_type> alpha_d(num_used_devices);
    #pragma omp parallel for default(none) shared(num_used_devices, devices_, alpha_d, alpha) firstprivate(num_support_vectors)
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        alpha_d[device] = device_ptr_type{ num_support_vectors + THREAD_BLOCK_SIZE, devices_[device] };
        alpha_d[device].memset(0);
        alpha_d[device].memcpy_to_device(alpha, 0, num_support_vectors);
    }

    std::vector<real_type> out(predict_points.size());

    // use faster methode in case of the linear kernel function
    if (params.kernel == kernel_type::linear && w.empty()) {
        w = calculate_w(data_d, data_last_d, alpha_d, support_vectors.size(), feature_ranges, num_used_devices);
    }

    if (params.kernel == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        #pragma omp parallel for default(none) shared(out, predict_points, w) firstprivate(num_predict_points, rho)
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < num_predict_points; ++i) {
            out[i] = transposed<real_type>{ w } * predict_points[i] + -rho;
        }
    } else {
        // create result vector on the device
        device_ptr_type out_d{ num_predict_points + boundary_size, devices_[0] };
        out_d.memset(0);

        // transform prediction data
        const std::vector<real_type> transformed_data = detail::transform_to_layout(detail::layout_type::soa, predict_points, boundary_size, predict_points.size());
        device_ptr_type point_d{ num_features * (num_predict_points + boundary_size), devices_[0] };
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


template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::run_device_kernel(const size_type device, const parameter<real_type> &params, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const device_ptr_type &data_d, const std::vector<size_type> &feature_ranges, const real_type QA_cost, const real_type add, const size_type dept, const size_type boundary_size) const {
    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept) / static_cast<real_type>(boundary_size)));
    const detail::execution_range range({ grid, grid }, { THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE });

    run_svm_kernel(device, range, params, q_d, r_d, x_d, data_d, QA_cost, add, dept + boundary_size, feature_ranges[device + 1] - feature_ranges[device]);
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::device_reduction(std::vector<device_ptr_type> &buffer_d, std::vector<real_type> &buffer) const {
    using namespace plssvm::operators;

    device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (buffer_d.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (typename std::vector<device_ptr_type>::size_type device = 1; device < buffer_d.size(); ++device) {
            device_synchronize(devices_[device]);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            buffer += ret;
        }

        #pragma omp parallel for default(none) shared(buffer_d, buffer)
        for (typename std::vector<device_ptr_type>::size_type device = 0; device < buffer_d.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::select_num_used_devices(const kernel_type kernel, const size_type num_features) const noexcept -> size_type {
    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel == kernel_type::polynomial || kernel == kernel_type::rbf) {
        std::clog << fmt::format("Warning: found {} devices, however only 1 device can be used since the polynomial and rbf kernels currently only support single GPU execution!", devices_.size()) << std::endl;
        return 1;
    }

    // the number of used devices may not exceed the number of features
    const size_type num_used_devices = std::min(devices_.size(), num_features);
    if (num_used_devices < devices_.size()) {
        std::clog << fmt::format("Warning: found {} devices, however only {} device(s) can be used since the data set only has {} features!", devices_.size(), num_used_devices, num_features) << std::endl;
    }
    return num_used_devices;
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
