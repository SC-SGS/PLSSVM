/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"  // plssvm::opencl::detail::device_ptr
#include "plssvm/backends/OpenCL/detail/error_code.hpp"  // plssvm::opencl::detail::error_code
#include "plssvm/backends/OpenCL/detail/utility.hpp"     // plssvm::opencl::detail::create_kernel, plssvm::opencl::detail::run_kernel
#include "plssvm/backends/OpenCL/exceptions.hpp"         // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                          // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                               // plssvm::csvm
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"                   // various operator overloads for std::vector and scalars
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all
#include "plssvm/exceptions/exceptions.hpp"              // plssvm::unsupported_kernel_type_exception
#include "plssvm/target_platform.hpp"                    // plssvm::target_platform

#include "CL/cl.h"     // clReleaseKernel
#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::min
#include <cmath>      // std::ceil
#include <string>     // std::string
#include <utility>    // std::pair
#include <vector>     // std::vector

namespace plssvm::opencl {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    ::plssvm::csvm<T>{ params } {
    // check whether the requested target platform has been enabled
    switch (target_) {
        case target_platform::automatic:
            break;
        case target_platform::cpu:
#if !defined(PLSSVM_HAS_CPU_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_nvidia:
#if !defined(PLSSVM_HAS_NVIDIA_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_amd:
#if !defined(PLSSVM_HAS_AMD_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
        case target_platform::gpu_intel:
#if !defined(PLSSVM_HAS_INTEL_TARGET)
            throw backend_exception{ fmt::format("Requested target platform {} that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", target_) };
#endif
            break;
    }

    if (print_info_) {
        fmt::print("Using OpenCL as backend.\n");
    }

    // TODO: check multi GPU
    // get all available devices wrt the requested target platform
    devices_ = detail::get_command_queues(target_);

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("OpenCL backend selected but no devices for the target {} were found!", target_) };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found OpenLC devices
        fmt::print("Found {} OpenCL device(s) for the target platform {}:\n", devices_.size(), target_);
        for (size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, detail::get_device_name(devices_[device]));
        }
        fmt::print("\n");
    }

    // get kernel names
    std::pair<std::string, std::string> kernel_names = detail::kernel_type_to_function_name(kernel_);
    // build necessary kernel
    q_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "q_kernel.cl", kernel_names.first);
    // assemble kernel name
    svm_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "svm_kernel.cl", kernel_names.second);

    switch (kernel_) {
        case kernel_type::linear:
            kernel_w_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict.cl", "kernel_w");
            break;
        case kernel_type::polynomial:
            predict_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict.cl", "predict_points_poly");
            break;
        case kernel_type::rbf:
            predict_kernel_ = detail::create_kernel<real_type, size_type>(devices_, PLSSVM_OPENCL_BACKEND_KERNEL_FILE_DIRECTORY "predict.cl", "predict_points_rbf");
            break;
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the OpenCL devices have finished before destruction
        for (const detail::command_queue &queue : devices_) {
            detail::device_synchronize(queue);
        }
    } catch (const plssvm::exception &e) {
        fmt::print("{}\n", e.what_with_loc());
        std::terminate();
    }
}

template <typename T>
void csvm<T>::setup_data_on_device() {
    // set values of member variables
    dept_ = num_data_points_ - 1;
    boundary_size_ = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    num_rows_ = static_cast<int>(dept_ + boundary_size_);
    num_cols_ = static_cast<int>(num_features_);

    // initialize data_last on device
    for (size_type device = 0; device < devices_.size(); ++device) {
        data_last_d_[device] = detail::device_ptr<real_type>{ num_features_ + boundary_size_, devices_[device] };
    }
#pragma omp parallel for
    for (size_type device = 0; device < devices_.size(); ++device) {
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_ptr_->back(), 0, num_features_);
    }

    // initialize data on devices
    for (size_type device = 0; device < devices_.size(); ++device) {
        data_d_[device] = detail::device_ptr<real_type>{ num_features_ * (dept_ + boundary_size_), devices_[device] };
    }
    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(*data_ptr_, boundary_size_, dept_);
#pragma omp parallel for
    for (size_type device = 0; device < devices_.size(); ++device) {
        data_d_[device].memcpy_to_device(transformed_data, 0, num_features_ * (dept_ + boundary_size_));
    }
}

template <typename T>
auto csvm<T>::generate_q() -> std::vector<real_type> {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<detail::device_ptr<real_type>> q_d(devices_.size());
    for (size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices_[device] };
        q_d[device].memset(0);
    }

    for (size_type device = 0; device < devices_.size(); ++device) {
        // feature splitting on multiple devices
        const int first_feature = static_cast<int>(device * static_cast<size_type>(num_cols_) / devices_.size());
        const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_cols_) / devices_.size());

        size_type block = std::min<size_type>(THREAD_BLOCK_SIZE, dept_);
        auto grid = static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE)) * block);

        switch (kernel_) {
            case kernel_type::linear:
                detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, first_feature, last_feature);
                break;
            case kernel_type::polynomial:
                detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_);
                break;
            case kernel_type::rbf:
                detail::run_kernel(devices_[device], q_kernel_[device], grid, block, q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_);
                break;
        }
    }

    std::vector<real_type> q(dept_);
    device_reduction(q_d, q);
    return q;
}

template <typename T>
void csvm<T>::run_device_kernel(const size_type device, const detail::device_ptr<real_type> &q_d, detail::device_ptr<real_type> &r_d, const detail::device_ptr<real_type> &x_d, const detail::device_ptr<real_type> &data_d, const real_type add) {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    // feature splitting on multiple devices
    const int first_feature = static_cast<int>(device * static_cast<size_type>(num_cols_) / devices_.size());
    const int last_feature = static_cast<int>((device + 1) * static_cast<size_type>(num_cols_) / devices_.size());

    const auto grid_dim = static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE)));
    std::vector<size_type> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };  // TODO: min?
    std::vector<size_type> grid{ grid_dim * block[0], grid_dim * block[1] };

    switch (kernel_) {
        case kernel_type::linear:
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, add, first_feature, last_feature);
            break;
        case kernel_type::polynomial:
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_);
            break;
        case kernel_type::rbf:
            detail::run_kernel(devices_[device], svm_kernel_[device], grid, block, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_);
            break;
    }
}

template <typename T>
void csvm<T>::device_reduction(std::vector<detail::device_ptr<real_type>> &buffer_d, std::vector<real_type> &buffer) {
    detail::device_synchronize(devices_[0]);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (devices_.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (size_type device = 1; device < devices_.size(); ++device) {
            detail::device_synchronize(devices_[device]);
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

template <typename T>
auto csvm<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<real_type> x(dept_, 1.0);
    std::vector<detail::device_ptr<real_type>> x_d(devices_.size());

    std::vector<real_type> r(dept_, 0.0);
    std::vector<detail::device_ptr<real_type>> r_d(devices_.size());

    for (size_type device = 0; device < devices_.size(); ++device) {
        x_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices_[device] };
        r_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices_[device] };
    }
#pragma omp parallel for
    for (size_type device = 0; device < devices_.size(); ++device) {
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept_);
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept_);

    std::vector<detail::device_ptr<real_type>> q_d(devices_.size());
    for (size_type device = 0; device < devices_.size(); ++device) {
        q_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices_[device] };
    }
#pragma omp parallel for
    for (size_type device = 0; device < devices_.size(); ++device) {
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept_);
    }

// r = Ax (r = b - Ax)
#pragma omp parallel for
    for (size_type device = 0; device < devices_.size(); ++device) {
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
    }

    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept_);

    std::vector<detail::device_ptr<real_type>> Ad_d(devices_.size());
    for (size_type device = 0; device < devices_.size(); ++device) {
        Ad_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices_[device] };
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
            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], data_d_[device], 1);
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
                run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
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

template <typename T>
void csvm<T>::update_w() {
    w_.resize(num_features_);
    w_d_ = detail::device_ptr<real_type>(num_features_ + THREAD_BLOCK_SIZE, devices_[0]);
    w_d_.memset(0);

    detail::device_ptr<real_type> alpha_d(num_data_points_ + THREAD_BLOCK_SIZE, devices_[0]);
    alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

    const std::vector<size_type> block{ std::min(THREAD_BLOCK_SIZE, static_cast<unsigned int>(num_features_)) };
    const std::vector<size_type> grid{ static_cast<unsigned int>(std::ceil(static_cast<real_type>(num_features_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) * block[0] };

    detail::run_kernel(devices_[0], kernel_w_kernel_[0], grid, block, w_d_.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, num_features_);
    detail::device_synchronize(devices_[0]);

    w_d_.memcpy_to_host(w_, 0, num_features_);
}

template <typename T>
auto csvm<T>::predict(const std::vector<std::vector<real_type>> &points) -> std::vector<real_type> {
    using namespace plssvm::operators;

    if (alpha_ptr_ == nullptr) {
        throw exception{ "No alphas provided for prediction!" };
    }

    PLSSVM_ASSERT(data_ptr_ != nullptr, "No data is provided!");
    PLSSVM_ASSERT(!data_ptr_->empty(), "Data set is empty!");
    PLSSVM_ASSERT(data_ptr_->size() == alpha_ptr_->size(), "Sizes mismatch!: {} != {}", data_ptr_->size(), alpha_ptr_->size());
    PLSSVM_ASSERT(!points.empty(), "No points to predict");
    for (const std::vector<real_type> &point : points) {
        PLSSVM_ASSERT(point.size() == num_features_, "Feature sizes mismatch!: {} != {}", point.size(), num_features_);
    }

    if (data_d_[0].empty()) {
        setup_data_on_device();
    }

    std::vector<real_type> out(points.size());

    if (kernel_ == kernel_type::linear) {
        // use faster methode in case of the linear kernel function
        if (w_.empty()) {
            update_w();
        }
        for (size_type i = 0; i < points.size(); ++i) {
            out[i] = transposed{ w_ } * points[i];
            out[i] += bias_;
        }
    } else {
        detail::device_ptr<real_type> out_d(points.size() + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE, devices_[0]);
        out_d.memset(0);

        std::vector<real_type> transformed_data = base_type::transform_data(points, THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE, points.size());
        detail::device_ptr<real_type> point_d(points[0].size() * (points.size() + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE), devices_[0]);
        point_d.memcpy_to_device(transformed_data, 0, transformed_data.size());

        detail::device_ptr<real_type> alpha_d(num_data_points_ + THREAD_BLOCK_SIZE, devices_[0]);
        alpha_d.memcpy_to_device(*alpha_ptr_.get(), 0, num_data_points_);

        std::vector<size_type> grid{ static_cast<unsigned int>(std::ceil(static_cast<real_type>(num_data_points_) / static_cast<real_type>(THREAD_BLOCK_SIZE)) * num_data_points_), static_cast<unsigned int>(std::ceil(static_cast<real_type>(points.size()) / static_cast<real_type>(THREAD_BLOCK_SIZE)) * points.size()) };
        std::vector<size_type> block{ std::min(THREAD_BLOCK_SIZE, static_cast<unsigned int>(num_data_points_)), std::min(THREAD_BLOCK_SIZE, static_cast<unsigned int>(points.size())) };

        switch (kernel_) {
            case kernel_type::linear:
                break;
            case kernel_type::polynomial:
                detail::run_kernel(devices_[0], predict_kernel_[0], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), points.size(), num_features_, degree_, gamma_, coef0_);
                break;
            case kernel_type::rbf:
                detail::run_kernel(devices_[0], predict_kernel_[0], grid, block, out_d.get(), data_d_[0].get(), data_last_d_[0].get(), alpha_d.get(), num_data_points_, point_d.get(), points.size(), num_features_, gamma_);
                break;
        }
        detail::device_synchronize(devices_[0]);

        out_d.memcpy_to_host(out, 0, points.size());
        for (size_type i = 0; i < points.size(); ++i) {
            out[i] += bias_;
        }
    }

    return out;
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::opencl
