/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/SYCL/csvm.hpp"

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"  // plssvm::detail::sycl::device_ptr, plssvm::detail::sycl::get_device_count, plssvm::detail::cuda::device_synchronize
#include "plssvm/backends/SYCL/exceptions.hpp"         // plssvm::sycl::backend_exception
#include "plssvm/backends/SYCL/q_kernel.hpp"           // plssvm::sycl::device_kernel_q_linear, plssvm::sycl::device_kernel_q_poly, plssvm::sycl::device_kernel_q_radial
#include "plssvm/backends/SYCL/svm_kernel.hpp"         // plssvm::sycl::device_kernel_linear, plssvm::sycl::device_kernel_poly, plssvm::sycl::device_kernel_radial
#include "plssvm/constants.hpp"                        // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                             // plssvm::csvm
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"                 // various operator overloads for std::vector and scalars
#include "plssvm/detail/utility.hpp"                   // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platform.hpp"                  // plssvm::target_platform

#include "fmt/core.h"     // fmt::print, fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::range, sycl::nd_range, sycl::handler

#include <algorithm>  // std::min
#include <cmath>      // std::ceil
#include <cstdio>     // stderr
#include <exception>  // std::exception, std::terminate
#include <vector>     // std::vector

namespace plssvm::sycl {

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
        fmt::print("Using SYCL as backend.\n");
    }

    // TODO: check multi GPU
    // get all available devices wrt the requested target platform
    devices_ = detail::get_device_list(target_);

    // throw exception if no devices for the requested target could be found
    if (devices_.empty()) {
        throw backend_exception{ fmt::format("SYCL backend selected but no devices for the target {} were found!", target_) };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        devices_.resize(1);
    }

    // resize vectors accordingly
    data_d_.resize(devices_.size());
    data_last_d_.resize(devices_.size());

    if (print_info_) {
        // print found SYCL devices
        fmt::print("Found {} SYCL device(s) for the target platform {}:\n", devices_.size(), target_);
        for (size_type device = 0; device < devices_.size(); ++device) {
            fmt::print("  [{}, {}]\n", device, devices_[device].get_device().template get_info<::sycl::info::device::name>());
        }
        fmt::print("\n");
    }
}

template <typename T>
csvm<T>::~csvm() {
    try {
        // be sure that all operations on the SYCL queues have finished before destruction
        for (::sycl::queue &q : devices_) {
            detail::device_synchronize(q);
        }
    } catch (const std::exception &e) {
        fmt::print("SYCL exception thrown: {}\n", e.what());
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

        const ::sycl::range<1> grid{ static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE))) };
        const ::sycl::range<1> block{ std::min<size_type>(THREAD_BLOCK_SIZE, dept_) };
        const ::sycl::nd_range<1> execution_range{ grid * block, block };

        switch (kernel_) {
            case kernel_type::linear:
                devices_[device].parallel_for(execution_range, device_kernel_q_linear{ q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, first_feature, last_feature });
                break;
            case kernel_type::polynomial:
                devices_[device].parallel_for(execution_range, device_kernel_q_poly{ q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, degree_, gamma_, coef0_ });
                break;
            case kernel_type::rbf:
                devices_[device].parallel_for(execution_range, device_kernel_q_radial{ q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, num_cols_, gamma_ });
                break;
        }
        detail::device_synchronize(devices_[device]);
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

    const auto grid_dim = static_cast<unsigned int>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(boundary_size_)));
    const ::sycl::range<2> grid{ grid_dim, grid_dim };
    const ::sycl::range<2> block{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };
    const ::sycl::nd_range<2> execution_range{ grid * block, block };

    switch (kernel_) {
        case kernel_type::linear:
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_linear{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, add, first_feature, last_feature });
            });
            break;
        case kernel_type::polynomial:
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_poly{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, degree_, gamma_, coef0_ });
            });
            break;
        case kernel_type::rbf:
            devices_[device].submit([&](::sycl::handler &cgh) {
                cgh.parallel_for(execution_range, device_kernel_radial{ cgh, q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, num_rows_, num_cols_, add, gamma_ });
            });
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

        // (r = b - A * x)
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

template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::sycl
