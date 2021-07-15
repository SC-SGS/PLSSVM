#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

#include "plssvm/backends/CUDA/CUDA_DevicePtr.cuh"   // plssvm::detail::cuda::device_ptr
#include "plssvm/backends/CUDA/CUDA_exceptions.hpp"  // plssvm::cuda_backend_exception
#include "plssvm/backends/CUDA/cuda-kernel.cuh"      // kernel_q
#include "plssvm/backends/CUDA/cuda-kernel.hpp"      // add_mult_
#include "plssvm/backends/CUDA/svm-kernel.cuh"       // kernel_linear, kernel_poly, kernel_radial
#include "plssvm/detail/operators.hpp"
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::copy, std::min
#include <cassert>    // assert
#include <cmath>      // std::ceil
#include <vector>     // std::vector

namespace plssvm {

using namespace plssvm::detail;

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(const parameter<T> &params) :
    CUDA_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info },
    num_devices_{ cuda::get_device_count() },
    data_d_(num_devices_),
    data_last_d_(num_devices_) {
    if (print_info_) {
        fmt::print("Using CUDA as backend.\n");
    }

    // throw exception if no CUDA devices could be found
    if (num_devices_ < 1) {
        throw cuda_backend_exception{ "CUDA backend selected but no CUDA devices were found!" };
    }

    if (print_info_) {
        // print found CUDA devices
        fmt::print("Found {} CUDA device(s):\n", num_devices_);
        for (int device = 0; device < num_devices_; ++device) {
            cudaDeviceProp prop;
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
        data_last_d_[device] = cuda::device_ptr<real_type>{ num_features_ + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_[num_data_points_ - 1], 0, num_features_);
    }

    // initialize data on devices
    for (int device = 0; device < num_devices_; ++device) {
        data_d_[device] = cuda::device_ptr<real_type>{ num_features_ * (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE), device };
    }
    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        data_d_[device].memcpy_to_device(transformed_data, 0, num_features_ * (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE));
    }
}

template <typename T>
auto CUDA_CSVM<T>::generate_q() -> std::vector<real_type> {
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    const int Ncols = num_features_;
    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;  // TODO: size_type?

    std::vector<cuda::device_ptr<real_type>> q_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::device_ptr<real_type>{ dept_all, device };
        q_d[device].memset(0);
    }

    for (int device = 0; device < num_devices_; ++device) {
        cuda::set_device(device);

        const int start = device * Ncols / num_devices_;
        const int end = (device + 1) * Ncols / num_devices_;
        const size_type grid = static_cast<size_type>(std::ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE)));
        const size_type block = std::min<size_type>(THREADBLOCK_SIZE, dept);

        kernel_q<<<grid, block>>>(q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), Nrows, start, end);

        cuda::peek_at_last_error();
    }

    std::vector<real_type> q(dept);
    cuda::device_synchronize(0);
    q_d[0].memcpy_to_host(q, 0, dept);
    std::vector<real_type> ret(dept);
    for (int device = 1; device < num_devices_; ++device) {
        cuda::device_synchronize(device);
        q_d[device].memcpy_to_host(ret, 0, dept);

        #pragma omp parallel for
        for (size_type i = 0; i < dept; ++i) {
            q[i] += ret[i];
        }
    }
    return q;
}

template <typename T>
void CUDA_CSVM<T>::run_device_kernel(const int device, const cuda::device_ptr<real_type> &q_d, cuda::device_ptr<real_type> &r_d, const cuda::device_ptr<real_type> &x_d, const cuda::device_ptr<real_type> &data_d, const int sign) {
    // TODO: size_type?
    const int Ncols = num_features_;
    const int Nrows = num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;

    const auto grid_dim = static_cast<unsigned int>(std::ceil(static_cast<real_type>(num_data_points_ - 1) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)));
    dim3 grid{ grid_dim, grid_dim };
    dim3 block{ static_cast<unsigned int>(THREADBLOCK_SIZE), static_cast<unsigned int>(THREADBLOCK_SIZE) };

    const int start = device * Ncols / num_devices_;
    const int end = (device + 1) * Ncols / num_devices_;

    switch (kernel_) {
        case kernel_type::linear:
            kernel_linear<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Ncols, Nrows, sign, start, end);
            break;
        case kernel_type::polynomial:
            kernel_poly<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Ncols, Nrows, sign, start, end, gamma_, coef0_, degree_);
            break;
        case kernel_type::rbf:
            kernel_radial<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost_, 1 / cost_, Ncols, Nrows, sign, start, end, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", detail::to_underlying(kernel_)) };
    }
}

template <typename T>
auto CUDA_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    // TODO: member variables!
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;

    std::vector<real_type> x(dept, 1.0);
    std::vector<cuda::device_ptr<real_type>> x_d(num_devices_);

    std::vector<real_type> r(dept, 0.0);
    std::vector<cuda::device_ptr<real_type>> r_d(num_devices_);

    for (int device = 0; device < num_devices_; ++device) {
        x_d[device] = cuda::device_ptr<real_type>{ dept_all, device };
        r_d[device] = cuda::device_ptr<real_type>{ dept_all, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept);
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept);

    std::vector<cuda::device_ptr<real_type>> q_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::device_ptr<real_type>{ dept_all, device };
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept);
    }

    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        cuda::set_device(device);
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
        cuda::peek_at_last_error();
    }

    cuda::device_synchronize(0);
    r_d[0].memcpy_to_host(r, 0, dept);
    {
        std::vector<real_type> ret(dept);
        for (int device = 1; device < num_devices_; ++device) {
            cuda::device_synchronize(device);
            r_d[device].memcpy_to_host(ret, 0, dept);

            #pragma omp parallel for
            for (size_type j = 0; j < dept; ++j) {
                r[j] += ret[j];
            }
        }
    }
    #pragma omp parallel for
    for (int device = 0; device < num_devices_; ++device) {
        r_d[device].memcpy_to_device(r, 0, dept);
    }

    real_type delta = mult(r, r);
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept);

    std::vector<cuda::device_ptr<real_type>> Ad_d(num_devices_);
    for (int device = 0; device < num_devices_; ++device) {
        Ad_d[device] = cuda::device_ptr<real_type>{ dept_all, device };
    }

    std::vector<real_type> d(r);

    size_type run;
    for (run = 0; run < imax; ++run) {
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
            cuda::set_device(device);
            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], data_d_[device], 1);
            cuda::peek_at_last_error();
        }

        // update Ad (q)
        cuda::device_synchronize(0);
        Ad_d[0].memcpy_to_host(Ad, 0, dept);
        {
            std::vector<real_type> ret(dept);
            for (int device = 1; device < num_devices_; ++device) {
                cuda::device_synchronize(device);
                Ad_d[device].memcpy_to_host(ret, 0, dept);

                #pragma omp parallel for
                for (size_type j = 0; j < dept; ++j) {
                    Ad[j] += ret[j];
                }
            }
        }
        // TODO: only if num_devices_ > 1?
        for (int device = 0; device < num_devices_; ++device) {
            Ad_d[device].memcpy_to_device(Ad, 0, dept);
        }

        // (alpha = delta_new / (d^T * q))
        real_type alpha_cd = delta / mult(d, Ad);

        // (x = x + alpha * d)
        add_mult_(((int) dept / 1024) + 1, std::min(1024, (int) dept), x.data(), d.data(), alpha_cd, dept);  // TODO: GPU (single <-> multi): add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
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
                cuda::set_device(device);
                run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], -1);
                cuda::peek_at_last_error();
            }

            cuda::device_synchronize(0);
            r_d[0].memcpy_to_host(r, 0, dept);
            {
                std::vector<real_type> ret(dept);
                for (int device = 1; device < num_devices_; ++device) {
                    cuda::device_synchronize(device);
                    r_d[device].memcpy_to_host(ret, 0, dept);

                    #pragma omp parallel for
                    for (size_type j = 0; j < dept; ++j) {
                        r[j] += ret[j];
                    }
                }
            }
            #pragma omp parallel for
            for (int device = 0; device < num_devices_; ++device) {
                r_d[device].memcpy_to_device(r, 0, dept);
            }
        } else {
            // r -= alpha_cd * Ad (r = r - alpha * q)
            for (size_type index = 0; index < dept; ++index) {
                r[index] -= alpha_cd * Ad[index];
            }
        }

        // (delta = r^T * r)
        real_type delta_old = delta;
        delta = mult(r, r);
        if (delta <= eps * eps * delta0) {
            break;
        }
        // (beta = delta_new / delta_old)
        real_type beta = delta / delta_old;
        // d = r + beta * d
        add(mult(beta, d), r, d);

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
