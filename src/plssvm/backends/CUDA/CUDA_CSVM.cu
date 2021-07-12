#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

#include "plssvm/backends/CUDA/CUDA_DevicePtr.cuh"  // plssvm::detail::cuda::device_ptr
#include "plssvm/backends/CUDA/cuda-kernel.cuh"     // kernel_q
#include "plssvm/backends/CUDA/cuda-kernel.hpp"     // add_mult_
#include "plssvm/backends/CUDA/svm-kernel.cuh"      // kernel_linear, kernel_poly, kernel_radial
#include "plssvm/detail/operators.hpp"
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_kernel_type_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::copy, std::min
#include <cmath>      // std::ceil
#include <vector>     // std::vector

namespace plssvm {

using namespace plssvm::detail;

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(parameter<T> &params) :
    CUDA_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info },
    num_devices_{ cuda::get_device_count() },
    data_d_(num_devices_),
    data_last_d_(num_devices_) {
    fmt::print("Found {} CUDA devices.", num_devices_);  // TODO: improve
}

template <typename T>
void CUDA_CSVM<T>::setup_data_on_device() {
    // initialize data_last on devices
    for (size_type device = 0; device < num_devices_; ++device) {
        data_last_d_[device] = cuda::device_ptr<real_type>{ num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE, static_cast<int>(device) };
    }
    std::vector<real_type> data_last(num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    std::copy(data_[num_data_points_ - 1].begin(), data_[num_data_points_ - 1].end(), data_last.begin());
    #pragma omp parallel for
    for (size_type device = 0; device < num_devices_; ++device) {
        data_last_d_[device].memcpy_to_device(data_last);
    }

    // initialize data on devices
    for (size_type device = 0; device < num_devices_; ++device) {
        data_d_[device] = cuda::device_ptr<real_type>{ num_features_ * (num_data_points_ + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE), static_cast<int>(device) };
    }
    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    #pragma omp parallel for
    for (size_type device = 0; device < num_devices_; ++device) {
        data_d_[device].memcpy_to_device(transformed_data, 0, num_features_ * (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE));  // TODO: look
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
    for (size_type device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::device_ptr<real_type>{ dept_all, static_cast<int>(device) };  // TODO: dept <-> dept_all
        q_d[device].memset(0);
    }
    cuda::device_synchronize();  // TODO: sync all devices?
    for (size_type device = 0; device < num_devices_; ++device) {
        cuda::set_device(device);

        const int start = device * Ncols / num_devices_;
        const int end = (device + 1) * Ncols / num_devices_;
        // TODO:
        kernel_q<<<((int) dept / THREADBLOCK_SIZE) + 1, std::min((size_type) THREADBLOCK_SIZE, dept)>>>(q_d[device].get(),
                                                                                                    data_d_[device].get(),
                                                                                                    data_last_d_[device].get(),
                                                                                                    Nrows,
                                                                                                    start,
                                                                                                    end);
        cuda::peek_at_last_error();
    }
    cuda::device_synchronize();

    std::vector<real_type> q(dept);
    //    cuda::set_device(0);
    q_d[0].memcpy_to_host(q, 0, dept);
    std::vector<real_type> ret(dept);  // TODO: dept_all vs dept?
    for (size_type device = 1; device < num_devices_; ++device) {
        q_d[device].memcpy_to_host(ret, 0, dept);
        for (size_type i = 0; i < dept; ++i) {
            q[i] += ret[i];
        }
    }
    return q;
}

template <typename T>
void CUDA_CSVM<T>::run_device_kernel(const size_type device, const cuda::device_ptr<real_type> &q_d, cuda::device_ptr<real_type> &r_d, const cuda::device_ptr<real_type> &x_d, const cuda::device_ptr<real_type> &data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int sign) {
    dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);
    dim3 grid(static_cast<size_type>(std::ceil(static_cast<real_type>(num_data_points_ - 1) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
              static_cast<size_type>(std::ceil(static_cast<real_type>(num_data_points_ - 1) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
    const int start = device * Ncols / num_devices_;
    const int end = (device + 1) * Ncols / num_devices_;
    switch (kernel_) {
        case kernel_type::linear:
            kernel_linear<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, cost, Ncols, Nrows, sign, start, end);
            break;
        case kernel_type::polynomial:
            kernel_poly<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, cost, Ncols, Nrows, sign, start, end, gamma_, coef0_, degree_);
            break;
        case kernel_type::rbf:
            kernel_radial<<<grid, block>>>(q_d.get(), r_d.get(), x_d.get(), data_d.get(), QA_cost, cost, Ncols, Nrows, sign, start, end, gamma_);
            break;
        default:
            throw unsupported_kernel_type_exception{ fmt::format("Unknown kernel type (value: {})!", static_cast<int>(kernel_)) };
    }
}

template <typename T>
auto CUDA_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    std::vector<real_type> zeros(dept_all, 0.0);

    // dim3 grid((int)dept/(THREADBLOCK_SIZE*INTERNALBLOCK_SIZE) + 1,(int)dept/(THREADBLOCK_SIZE*INTERNALBLOCK_SIZE) + 1);
    //    dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);

    std::vector<real_type> x(dept_all, 1.0);
    std::fill(x.end() - boundary_size, x.end(), 0.0);

    std::vector<cuda::device_ptr<real_type>> x_d(num_devices_);
    std::vector<real_type> r(dept_all, 0.0);
    std::vector<cuda::device_ptr<real_type>> r_d(num_devices_);
    for (size_type device = 0; device < num_devices_; ++device) {
        x_d[device] = cuda::device_ptr<real_type>{ dept_all, static_cast<int>(device) };
        x_d[device].memcpy_to_device(x);
        r_d[device] = cuda::device_ptr<real_type>{ dept_all, static_cast<int>(device) };
    }

    r_d[0].memcpy_to_device(b, 0, dept);
    r_d[0].memset(0, dept);
    #pragma omp parallel for
    for (size_type device = 1; device < num_devices_; ++device) {
        r_d[device].memset(0);
    }
    std::vector<real_type> d(dept);

    // TODO: size_type
    const int Ncols = num_features_;
    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    cuda::device_synchronize();

    std::vector<cuda::device_ptr<real_type>> q_d(num_devices_);
    for (size_type device = 0; device < num_devices_; ++device) {
        q_d[device] = cuda::device_ptr<real_type>{ dept_all, static_cast<int>(device) };
        //        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept);
        //        q_d[device].memcpy_to_device(q, 0, dept_all);  // TODO:
    }
    cuda::device_synchronize();

    #pragma omp parallel for
    for (size_type device = 0; device < num_devices_; ++device) {
        cuda::set_device(device);
        run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], QA_cost_, 1 / cost_, Ncols, Nrows, -1);
        cuda::peek_at_last_error();
    }

    // cudaMemcpy(r, r_d, dept*sizeof(real_type), cudaMemcpyDeviceToHost);
    cuda::device_synchronize();
    {
        r_d[0].memcpy_to_host(r);
        std::vector<real_type> ret(dept_all);
        for (int device = 1; device < num_devices_; ++device) {
            r_d[device].memcpy_to_host(ret);
            for (size_type j = 0; j <= dept; ++j) {
                r[j] += ret[j];
            }
        }
    }
    real_type delta = mult(r.data(), r.data(), dept);  // TODO:
    const real_type delta0 = delta;
    real_type alpha_cd;
    real_type beta;
    std::vector<real_type> Ad(dept);

    std::vector<cuda::device_ptr<real_type>> Ad_d(num_devices_);
    for (size_type device = 0; device < num_devices_; ++device) {
        Ad_d[device] = cuda::device_ptr<real_type>{ dept_all, static_cast<int>(device) };
        Ad_d[device].memcpy_to_device(r);
    }
    // cudaMallocHost((void **) &Ad, dept *sizeof(real_type));

    size_type run;
    for (run = 0; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }
        // Ad = A * d
        for (size_type device = 0; device < num_devices_; ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept);
        }

        #pragma omp parallel for
        for (size_type device = 0; device < num_devices_; ++device) {
            cuda::set_device(device);
            run_device_kernel(device, q_d[device], Ad_d[device], r_d[device], data_d_[device], QA_cost_, 1 / cost_, Ncols, Nrows, 1);
            cuda::peek_at_last_error();
        }

        for (size_type i = 0; i < dept; ++i) {
            d[i] = r[i];
        }

        cuda::device_synchronize();
        {
            std::vector<real_type> buffer(dept_all, 0);
            std::vector<real_type> ret(dept_all);
            for (size_type device = 0; device < num_devices_; ++device) {
                Ad_d[device].memcpy_to_host(ret);
                for (size_type j = 0; j <= dept; ++j) {
                    buffer[j] += ret[j];
                }
            }
            std::copy(buffer.begin(), buffer.begin() + dept, Ad.data());
            for (size_type device = 0; device < num_devices_; ++device) {
                Ad_d[device].memcpy_to_device(buffer);
            }
        }

        alpha_cd = delta / mult(d.data(), Ad.data(), dept);
        // add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
        // TODO: move to GPU
        std::vector<real_type> buffer_r(dept_all);
        r_d[0].memcpy_to_host(buffer_r);
        add_mult_(((int) dept / 1024) + 1, std::min(1024, (int) dept), x.data(), buffer_r.data(), alpha_cd, dept);

        #pragma omp parallel for
        for (size_type device = 0; device < num_devices_; ++device) {
            x_d[device].memcpy_to_device(x);
        }
        if (run % 50 == 49) {
            std::vector<real_type> buffer(dept_all);
            std::copy(b.begin(), b.end(), buffer.begin());  // TODO:
                                                            //            std::vector<real_type> buffer(b);
                                                            //            buffer.resize(dept_all);
            r_d[0].memcpy_to_device(buffer);
            #pragma omp parallel for
            for (size_type device = 1; device < num_devices_; ++device) {
                r_d[device].memset(0);
            }

            #pragma omp parallel for
            for (size_type device = 0; device < num_devices_; ++device) {
                cuda::set_device(device);
                run_device_kernel(device, q_d[device], r_d[device], x_d[device], data_d_[device], QA_cost_, 1 / cost_, Ncols, Nrows, -1);
                cuda::peek_at_last_error();
            }
            cuda::device_synchronize();
            // cudaMemcpy(r, r_d, dept*sizeof(real_type), cudaMemcpyDeviceToHost);

            {
                r_d[0].memcpy_to_host(r);
                std::vector<real_type> ret(dept_all, 0);
                //                #pragma omp parallel for // TODO: race conditions?!
                for (size_type device = 1; device < num_devices_; ++device) {
                    r_d[device].memcpy_to_host(ret);
                    ;
                    for (size_type j = 0; j <= dept; ++j) {
                        r[j] += ret[j];
                    }
                }
                #pragma omp parallel for
                for (size_type device = 0; device < num_devices_; ++device) {
                    r_d[device].memcpy_to_device(r);
                }
            }
        } else {
            for (size_type index = 0; index < dept; ++index) {
                r[index] -= alpha_cd * Ad[index];
            }
        }

        delta = mult(r.data(), r.data(), dept);  // TODO:
        if (delta < eps * eps * delta0) {
            break;
        }
        beta = -mult(r.data(), Ad.data(), dept) / mult(d.data(), Ad.data(), dept);  // TODO:
        add(mult(beta, d.data(), dept), r.data(), d.data(), dept);                  // TODO:

        {
            std::vector<real_type> buffer(dept_all, 0);
            std::copy(d.begin(), d.begin() + dept, buffer.begin());
            #pragma omp parallel for
            for (size_type device = 0; device < num_devices_; ++device) {
                r_d[device].memcpy_to_device(buffer);
            }
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    alpha_.resize(dept);
    std::vector<real_type> ret_q(dept);
    cuda::device_synchronize();
    {
        std::vector<real_type> buffer(dept_all);
        std::copy(x.begin(), x.begin() + dept, alpha_.begin());
        q_d[0].memcpy_to_host(buffer);
        std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
    }
    // cudaMemcpy(&alpha[0],x_d, dept * sizeof(real_type), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&ret_q[0],q_d, dept * sizeof(real_type), cudaMemcpyDeviceToHost);
    return alpha_;
}

template class CUDA_CSVM<float>;
template class CUDA_CSVM<double>;

}  // namespace plssvm
