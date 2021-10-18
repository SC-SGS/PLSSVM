#include "plssvm/backends/gpu_csvm.hpp"

#include "plssvm/constants.hpp"               // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                    // plssvm::csvm
#include "plssvm/detail/execution_range.hpp"  // plssvm::detail::execution_range
#include "plssvm/detail/operators.hpp"        // various operator overloads for std::vector and scalars
#include "plssvm/parameter.hpp"               // plssvm::parameter

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    // used for explicitly instantiating the CUDA version
    #include "plssvm/backends/CUDA/detail/device_ptr.cuh"
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    // used for explicitly instantiating the OpenCL version
    #include "plssvm/backends/OpenCL/detail/command_queue.hpp"
    #include "plssvm/backends/OpenCL/detail/device_ptr.hpp"
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // used for explicitly instantiating the SYCL version
    #include "plssvm/backends/SYCL/detail/device_ptr.hpp"
    #include "sycl/sycl.hpp"
#endif

#include <algorithm>  // std::all_of, std::min
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <vector>     // std::vector

#include <iostream>
namespace plssvm::detail {

template <typename T, typename device_ptr_t, typename queue_t>
gpu_csvm<T, device_ptr_t, queue_t>::gpu_csvm(const parameter<T> &params) :
    base_type{ params } {}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::predict(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");  // exception in constructor
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");     // exception in constructor

    // return empty vector if there are no points to predict
    if (points.empty()) {
        return std::vector<real_type>{};
    }

    // sanity checks
    if (!std::all_of(points.begin(), points.end(), [&](const std::vector<real_type> &point) { return point.size() == points.front().size(); })) {
        throw exception{ "All points in the prediction point vector must have the same number of features!" };
    } else if (points.front().size() != data_ptr_->front().size()) {
        throw exception{ fmt::format("Number of features per data point ({}) must match the number of features per predict point ({})!", data_ptr_->front().size(), points.front().size()) };
    } else if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas provided for prediction!" };
    }

    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());  // exception in constructor

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
        for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < points.size(); ++i) {
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

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_data_points_) / static_cast<real_type>(THREAD_BLOCK_SIZE))),
                                              static_cast<std::size_t>(std::ceil(static_cast<real_type>(points.size()) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_data_points_), std::min<std::size_t>(THREAD_BLOCK_SIZE, points.size()) });

        // perform prediction on the first device
        run_predict_kernel(range, out_d, alpha_d, point_d, points.size());

        out_d.memcpy_to_host(out, 0, points.size());

        // add bias_ to all predictions
        out += bias_;
    }

    return out;
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::setup_data_on_device() {
    // set values of member variables
    dept_ = num_data_points_ - 1;
    boundary_size_ = static_cast<std::size_t>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    num_rows_ = dept_ + boundary_size_;
    num_cols_ = num_features_;
    feature_ranges_.reserve(devices_.size() + 1);
    for (typename std::vector<queue_type>::size_type device = 0; device <= devices_.size(); ++device) {
        feature_ranges_.push_back(device * num_cols_ / devices_.size());
    }

    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(*data_ptr_, boundary_size_, dept_);

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        const std::size_t num_features = feature_ranges_[device + 1] - feature_ranges_[device];

        // initialize data_last on device
        data_last_d_[device] = device_ptr_type{ num_features + boundary_size_, devices_[device] };
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_ptr_->back().data() + feature_ranges_[device], 0, num_features);

        const std::size_t device_data_size = num_features * (dept_ + boundary_size_);
        data_d_[device] = device_ptr_type{ device_data_size, devices_[device] };
        data_d_[device].memcpy_to_device(transformed_data.data() + feature_ranges_[device] * (dept_ + boundary_size_), 0, device_data_size);
    }
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::generate_q() -> std::vector<real_type> {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<device_ptr_type> q_d(devices_.size());

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);

        // feature splitting on multiple devices
        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_) });

        run_q_kernel(device, range, q_d[device], feature_ranges_[device + 1] - feature_ranges_[device]);
    }

    std::vector<real_type> q(dept_);
    device_reduction(q_d, q);
    return q;
}

template <typename T, typename device_ptr_t, typename queue_t>
auto gpu_csvm<T, device_ptr_t, queue_t>::solver_CG(const std::vector<real_type> &b, const std::size_t imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<real_type> x(dept_, 1.0);
    std::vector<device_ptr_type> x_d(devices_.size());

    std::vector<real_type> r(dept_, 0.0);
    std::vector<device_ptr_type> r_d(devices_.size());

    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        x_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept_);

        r_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept_);

    std::vector<device_ptr_type> q_d(devices_.size());
    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = device_ptr_type{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept_);

        // r = Ax (r = b - Ax)
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], -1);
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

    std::size_t run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }
        // Ad = A * r (q = A * d)
        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept_);

            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], 1);
        }
        // update Ad (q)
        device_reduction(Ad_d, Ad);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            x_d[device].memcpy_to_device(x, 0, dept_);
        }

        if (run % 50 == 49) {
            // r = b
            r_d[0].memcpy_to_device(b, 0, dept_);
            #pragma omp parallel for
            for (typename std::vector<queue_type>::size_type device = 1; device < devices_.size(); ++device) {
                r_d[device].memset(0);

                // r -= A * x
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
        const real_type beta = delta / delta_old;
        // d = beta * d + r
        d = beta * d + r;

        // r_d = d
        #pragma omp parallel for
        for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
            r_d[device].memcpy_to_device(d, 0, dept_);
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    return std::vector<real_type>(x.begin(), x.begin() + dept_);
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::update_w() {
    w_.resize(num_features_);
    #pragma omp parallel for
    for (typename std::vector<queue_type>::size_type device = 0; device < devices_.size(); ++device) {
        // feature splitting on multiple devices
        const std::size_t num_features = feature_ranges_[device + 1] - feature_ranges_[device];

        // create the w vector on the device
        device_ptr_type w_d = device_ptr_type{ num_features, devices_[device] };
        // create the weight vector on the device and copy data
        device_ptr_type alpha_d{ num_data_points_ + THREAD_BLOCK_SIZE, devices_[device] };
        alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

        const detail::execution_range range({ static_cast<std::size_t>(std::ceil(static_cast<real_type>(num_features) / static_cast<real_type>(THREAD_BLOCK_SIZE))) },
                                            { std::min<std::size_t>(THREAD_BLOCK_SIZE, num_features) });

        // calculate the w vector on the device
        run_w_kernel(device, range, w_d, alpha_d, num_features);
        device_synchronize(devices_[device]);

        // copy back to host memory
        w_d.memcpy_to_host(w_.data() + feature_ranges_[device], 0, num_features);
    }
}

template <typename T, typename device_ptr_t, typename queue_t>
void gpu_csvm<T, device_ptr_t, queue_t>::run_device_kernel(const std::size_t device, const device_ptr_type &q_d, device_ptr_type &r_d, const device_ptr_type &x_d, const real_type add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    const auto grid = static_cast<std::size_t>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const auto block = std::min<std::size_t>(THREAD_BLOCK_SIZE, dept_);
    const detail::execution_range range({ grid, grid }, { block, block });

    run_svm_kernel(device, range, q_d, r_d, x_d, add, feature_ranges_[device + 1] - feature_ranges_[device]);
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

        #pragma omp parallel for
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
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
template class gpu_csvm<float, ::plssvm::opencl::detail::device_ptr<float>, ::plssvm::opencl::detail::command_queue>;
template class gpu_csvm<double, ::plssvm::opencl::detail::device_ptr<double>, ::plssvm::opencl::detail::command_queue>;
#endif
#if defined(PLSSVM_HAS_SYCL_BACKEND)
template class gpu_csvm<float, ::plssvm::sycl::detail::device_ptr<float>, ::sycl::queue>;
template class gpu_csvm<double, ::plssvm::sycl::detail::device_ptr<double>, ::sycl::queue>;
#endif

}  // namespace plssvm::detail