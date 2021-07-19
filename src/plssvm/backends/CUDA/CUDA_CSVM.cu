/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::detail::cuda::device_ptr, plssvm::detail::cuda::get_device_count, plssvm::detail::cuda::set_device,
                                                       // plssvm::detail::cuda::peek_at_last_error, plssvm::detail::cuda::device_synchronize
#include "plssvm/backends/CUDA/exceptions.hpp"         // plssvm::cuda_backend_exception
#include "plssvm/backends/CUDA/q-kernel.cuh"           // plssvm::cuda::kernel_q_linear, plssvm::cuda::kernel_q_poly, plssvm::cuda::kernel_q_radial
#include "plssvm/backends/CUDA/svm-kernel.cuh"         // plssvm::cuda::kernel_linear, plssvm::cuda::kernel_poly, plssvm::cuda::kernel_radial
#include "plssvm/detail/operators.hpp"                 // various operator overloads for std::vector and scalars
#include "plssvm/detail/utility.hpp"                   // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"            // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/typedef.hpp"                          // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::min
#include <cmath>      // std::ceil
#include <vector>     // std::vector

namespace plssvm {

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(const parameter<T> &params) :
    CUDA_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info },
    num_devices_{ cuda::detail::get_device_count() },
    data_d_(num_devices_),
    data_last_d_(num_devices_) {
    if (print_info_) {
        fmt::print("Using CUDA as backend.\n");
    }

    // throw exception if no CUDA devices could be found
    if (num_devices_ < 1) {
        throw cuda_backend_exception{ "CUDA backend selected but no CUDA devices were found!" };
    }

    // polynomial and rbf kernel currently only support single GPU execution
    if (kernel_ == kernel_type::polynomial || kernel_ == kernel_type::rbf) {
        num_devices_ = 1;
    }

    if (print_info_) {
        // print found CUDA devices
        fmt::print("Found {} CUDA device(s):\n", num_devices_);
        for (int device = 0; device < num_devices_; ++device) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            fmt::print("  [{}, {}, {}.{}]\n", device, prop.name, prop.major, prop.minor);
        }
        fmt::print("\n");
    }
}

template <typename T>
void CUDA_CSVM<T>::setup_data_on_device() {
    // initialize data_last on devices
    for (int device = 0; device < num_devices_; ++device) {
        data_last_d_[device] = cuda::detail::device_ptr<real_type>{ num_features_ + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_[num_data_points_ - 1], 0, num_features_);
    }

    // initialize data on devices
    for (int device = 0; device < num_devices_; ++device) {
        data_d_[device] = cuda::detail::device_ptr<real_type>{ num_features_ * (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE), device };
    }
    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        data_d_[device].memcpy_to_device(transformed_data, 0, num_features_ * (num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE));
    }
}

template <typename T>
auto CUDA_CSVM<T>::generate_q() -> std::vector<real_type> {
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    const int Ncols = num_features_;
    const int Nrows = dept + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;  // TODO: size_type?

    std::vector<cuda::detail::device_ptr<real_type>> q_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::detail::device_ptr<real_type>{ dept_all, device };
        q_d[device].memset(0);
    }

    for (int device = 0; device < num_devices_; ++device) {
        cuda::detail::set_device(device);

        const int start = device * Ncols / num_devices_;
        const int end = (device + 1) * Ncols / num_devices_;
        const size_type grid = static_cast<size_type>(std::ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREAD_BLOCK_SIZE)));
        const size_type block = std::min<size_type>(THREAD_BLOCK_SIZE, dept);

        switch (kernel_) {
            case kernel_type::linear:
                cuda::kernel_q_linear<<<grid, block>>>(q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), Nrows, start, end);
                break;
            case kernel_type::polynomial:
                cuda::kernel_q_poly<<<grid, block>>>(q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), Nrows, Ncols, degree_, gamma_, coef0_);
                break;
            case kernel_type::rbf:
                cuda::kernel_q_radial<<<grid, block>>>(q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), Nrows, Ncols, gamma_);
                break;
        }

        cuda::detail::peek_at_last_error();
    }

    std::vector<real_type> q(dept);
    device_reduction(q_d, q);
    return q;
}

template <typename T>
void CUDA_CSVM<T>::run_device_kernel(const int device, const cuda::detail::device_ptr<real_type> &q_d, cuda::detail::device_ptr<real_type> &r_d, const cuda::detail::device_ptr<real_type> &x_d, const cuda::detail::device_ptr<real_type> &data_d, const int add) {
    // TODO: size_type?
    const int Ncols = num_features_;
    const int Nrows = num_data_points_ - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;

    const auto grid_dim = static_cast<unsigned int>(std::ceil(static_cast<real_type>(num_data_points_ - 1) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE)));
    dim3 grid{ grid_dim, grid_dim };
    dim3 block{ static_cast<unsigned int>(THREAD_BLOCK_SIZE), static_cast<unsigned int>(THREAD_BLOCK_SIZE) };

    const int start = device * Ncols / num_devices_;
    const int end = (device + 1) * Ncols / num_devices_;

    switch (kernel_) {
        case kernel_type::linear:
            cuda::kernel_linear<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Nrows, add, start, end);
            break;
        case kernel_type::polynomial:
            cuda::kernel_poly<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Ncols, Nrows, add, gamma_, coef0_, degree_);
            break;
        case kernel_type::rbf:
            cuda::kernel_radial<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Ncols, Nrows, add, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(kernel_)) };
    }
}

template <typename T>
void CUDA_CSVM<T>::device_reduction(std::vector<cuda::detail::device_ptr<real_type>> &buffer_d, std::vector<real_type> &buffer) {
    cuda::detail::device_synchronize(0);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (num_devices_ > 1) {
        std::vector<real_type> ret(buffer.size());
        for (int device = 1; device < num_devices_; ++device) {
            cuda::detail::device_synchronize(device);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            #pragma omp parallel for
            for (size_type j = 0; j < ret.size(); ++j) {
                buffer[j] += ret[j];
            }
        }

        #pragma omp parallel for
        for (int device = 0; device < num_devices_; ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

template <typename T>
auto CUDA_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    // TODO: member variables!
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;

    std::vector<real_type> x(dept, 1.0);
    std::vector<cuda::detail::device_ptr<real_type>> x_d(num_devices_);

    std::vector<real_type> r(dept, 0.0);
    std::vector<cuda::detail::device_ptr<real_type>> r_d(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {
        x_d[device] = cuda::detail::device_ptr<real_type>{ dept_all, device };
        r_d[device] = cuda::detail::device_ptr<real_type>{ dept_all, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept);
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept);

    std::vector<cuda::detail::device_ptr<real_type>> q_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::detail::device_ptr<real_type>{ dept_all, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept);
    }

    // r = Ax (r = b - Ax)
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        cuda::detail::set_device(device);
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
        cuda::detail::peek_at_last_error();
    }

    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<cuda::detail::device_ptr<real_type>> Ad_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        Ad_d[device] = cuda::detail::device_ptr<real_type>{ dept_all, device };
    }

    std::vector<real_type> d(r);

    size_type run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }

        // Ad = A * r (q = A * d)
        #pragma omp parallel for
        for (int device = 0; device < num_devices_; ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept);
        }
        #pragma omp parallel for
        for (int device = 0; device < num_devices_; ++device) {
            cuda::detail::set_device(device);
            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], data_d_[device], 1);
            cuda::detail::peek_at_last_error();
        }

        // update Ad (q)
        device_reduction(Ad_d, Ad);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

        #pragma omp parallel for
        for (int device = 0; device < num_devices_; ++device) {
            x_d[device].memcpy_to_device(x, 0, dept);
        }

        // (r = b - A * x)
        if (run % 50 == 49) {
            // r = b
            r_d[0].memcpy_to_device(b, 0, dept);
            #pragma omp parallel for
            for (int device = 1; device < num_devices_; ++device) {
                r_d[device].memset(0);
            }

            // r -= A * x
            #pragma omp parallel for
            for (int device = 0; device < num_devices_; ++device) {
                cuda::detail::set_device(device);
                run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
                cuda::detail::peek_at_last_error();
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
        for (int device = 0; device < num_devices_; ++device) {
            r_d[device].memcpy_to_device(d, 0, dept);
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    alpha_.assign(x.begin(), x.begin() + dept);
    // alpha_.resize(dept);
    // x_d[0].memcpy_to_host(alpha_, 0, dept);

    return alpha_;
}

template class CUDA_CSVM<float>;
template class CUDA_CSVM<double>;

}  // namespace plssvm
