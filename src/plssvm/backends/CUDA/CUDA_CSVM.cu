#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"
#include "plssvm/backends/CUDA/cuda-kernel.cuh"
#include "plssvm/backends/CUDA/cuda-kernel.hpp"
#include "plssvm/backends/CUDA/svm-kernel.cuh"
#include "plssvm/detail/operators.hpp"

#include <chrono>

namespace plssvm {

int CUDADEVICE = 0;

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

int count_devices = 1;

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(parameter<T> &params) :
    CUDA_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
CUDA_CSVM<T>::CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    gpuErrchk(cudaGetDeviceCount(&count_devices));
    datlast_d = std::vector<real_type *>(count_devices);
    data_d = std::vector<real_type *>(count_devices);

    std::cout << "GPUs found: " << count_devices << std::endl;
}

template <typename T>
void CUDA_CSVM<T>::setup_data_on_device() {
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &datlast_d[device],
                             (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_type)));
    }
    std::vector<real_type> datalast(data_[num_data_points_ - 1]);
    datalast.resize(num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMemcpy(datlast_d[device], datalast.data(), (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_type), cudaMemcpyHostToDevice));
    }
    datalast.resize(num_data_points_ - 1);
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &data_d[device],
                             num_features_ * (num_data_points_ + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_type)));
    }

    auto begin_transform = std::chrono::high_resolution_clock::now();
    const std::vector<real_type> transformet_data = base_type::transform_data(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    auto end_transform = std::chrono::high_resolution_clock::now();
    if (print_info_) {
        std::clog << std::endl
                  << data_.size() << " Datenpunkte mit Dimension " << num_features_ << " in "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end_transform - begin_transform).count()
                  << " ms transformiert" << std::endl;
    }
    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));

        gpuErrchk(cudaMemcpy(data_d[device], transformet_data.data(), num_features_ * (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_type), cudaMemcpyHostToDevice));
    }
}

template <typename T>
auto CUDA_CSVM<T>::generate_q() -> std::vector<real_type> {
    if (print_info_) {
        std::cout << "kernel_q" << std::endl;
    }

    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    const int Ncols = num_features_;
    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;

    std::vector<real_type *> q_d(count_devices);
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &q_d[device], dept_all * sizeof(real_type)));
        gpuErrchk(cudaMemset(q_d[device], 0, dept_all * sizeof(real_type)));
    }
    gpuErrchk(cudaDeviceSynchronize());
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));

        const int start = device * Ncols / count_devices;
        const int end = (device + 1) * Ncols / count_devices;
        kernel_q<<<((int) dept / CUDABLOCK_SIZE) + 1, std::min((size_type) CUDABLOCK_SIZE, dept)>>>(q_d[device],
                                                                                                    data_d[device],
                                                                                                    datlast_d[device],
                                                                                                    Nrows,
                                                                                                    start,
                                                                                                    end);
        gpuErrchk(cudaPeekAtLastError());
    }
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<real_type> q(dept);
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaMemcpy(q.data(), q_d[0], dept * sizeof(real_type), cudaMemcpyDeviceToHost));
    std::vector<real_type> ret(dept_all);
    for (size_type device = 1; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMemcpy(ret.data(), q_d[device], dept * sizeof(real_type), cudaMemcpyDeviceToHost));
        for (size_type i = 0; i < dept; ++i) {
            q[i] += ret[i];
        }
    }
    return q;
}

template <typename T>
auto CUDA_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    std::vector<real_type> zeros(dept_all, 0.0);

    // dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
    dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);

    real_type *d;
    std::vector<real_type> x(dept_all, 1.0);
    std::fill(x.end() - boundary_size, x.end(), 0.0);

    std::vector<real_type *> x_d(count_devices);
    std::vector<real_type> r(dept_all, 0.0);
    std::vector<real_type *> r_d(count_devices);
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &x_d[device], dept_all * sizeof(real_type)));
        gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc((void **) &r_d[device], dept_all * sizeof(real_type)));
    }

    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaMemcpy(r_d[0], b.data(), dept * sizeof(real_type), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(r_d[0] + dept, 0, (dept_all - dept) * sizeof(real_type)));
    #pragma omp parallel for
    for (size_type device = 1; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMemset(r_d[device], 0, dept_all * sizeof(real_type)));
    }
    d = new real_type[dept];

    const int Ncols = num_features_;
    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<real_type *> q_d(count_devices);
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &q_d[device], dept_all * sizeof(real_type)));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemset(q_d[device], 0, dept_all * sizeof(real_type)));
        gpuErrchk(cudaDeviceSynchronize());
        gpuErrchk(cudaMemcpy(q_d[device], q.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
    }

    gpuErrchk(cudaDeviceSynchronize());

    switch (kernel_) {
        case kernel_type::linear: {
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                dim3 grid(static_cast<size_type>(ceil(
                              static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                          static_cast<size_type>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
                const int start = device * Ncols / count_devices;
                const int end = (device + 1) * Ncols / count_devices;
                kernel_linear<<<grid, block>>>(q_d[device], r_d[device], x_d[device], data_d[device], QA_cost_, 1 / cost_, Ncols, Nrows, -1, start, end);
                gpuErrchk(cudaPeekAtLastError());
            }
            break;
        }
        case kernel_type::polynomial:
            // kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost_, 1/cost, num_features_ , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma_, coef0_, degree_);
            break;
        case kernel_type::rbf:
            // kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost_, 1/cost_, num_features_ , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma_);
            break;
        default:
            throw std::runtime_error("Can not decide which kernel!");
    }

    // cudaMemcpy(r, r_d, dept*sizeof(real_type), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaDeviceSynchronize());
    {
        gpuErrchk(cudaSetDevice(0));
        gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
        for (int device = 1; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            std::vector<real_type> ret(dept_all);
            gpuErrchk(cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
            for (size_type j = 0; j <= dept; ++j) {
                r[j] += ret[j];
            }
        }
    }
    real_type delta = mult(r.data(), r.data(), dept);  // TODO:
    const real_type delta0 = delta;
    real_type alpha_cd, beta;
    std::vector<real_type> Ad(dept);

    std::vector<real_type *> Ad_d(count_devices);
    for (size_type device = 0; device < count_devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaMalloc((void **) &Ad_d[device], dept_all * sizeof(real_type)));
        gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
    }
    //cudaMallocHost((void **) &Ad, dept *sizeof(real_type));

    size_type run;
    for (run = 0; run < imax; ++run) {
        if (print_info_) {
            std::cout << "Start Iteration: " << run << std::endl;
        }
        //Ad = A * d
        for (size_type device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMemset(Ad_d[device], 0, dept_all * sizeof(real_type)));
            gpuErrchk(cudaMemset(r_d[device] + dept, 0, (dept_all - dept) * sizeof(real_type)));
        }
        switch (kernel_) {
            case kernel_type::linear: {
                #pragma omp parallel for
                for (size_type device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    dim3 grid(static_cast<size_type>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                              static_cast<size_type>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
                    const int start = device * Ncols / count_devices;
                    const int end = (device + 1) * Ncols / count_devices;
                    kernel_linear<<<grid, block>>>(q_d[device], Ad_d[device], r_d[device], data_d[device], QA_cost_, 1 / cost_, Ncols, Nrows, 1, start, end);
                    gpuErrchk(cudaPeekAtLastError());
                }
            } break;
            case kernel_type::polynomial:
                // kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost_, 1/cost_, num_features_, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma_, coef0_, degree_);
                break;
            case kernel_type::rbf:
                // kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost_, 1/cost_, num_features_, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma_);
                break;
            default:
                throw std::runtime_error("Can not decide which kernel!");
        }

        for (size_type i = 0; i < dept; ++i) {
            d[i] = r[i];
        }

        gpuErrchk(cudaDeviceSynchronize());
        {
            std::vector<real_type> buffer(dept_all, 0);
            for (size_type device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                std::vector<real_type> ret(dept_all, 0);
                gpuErrchk(cudaMemcpy(ret.data(), Ad_d[device], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
                for (size_type j = 0; j <= dept; ++j) {
                    buffer[j] += ret[j];
                }
            }
            std::copy(buffer.begin(), buffer.begin() + dept, Ad.data());
            for (size_type device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(
                    cudaMemcpy(Ad_d[device], buffer.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
            }
        }

        alpha_cd = delta / mult(d, Ad.data(), dept);
        // add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
        //TODO: auf GPU
        std::vector<real_type> buffer_r(dept_all);
        cudaSetDevice(0);
        gpuErrchk(cudaMemcpy(buffer_r.data(), r_d[0], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
        add_mult_(((int) dept / 1024) + 1, std::min(1024, (int) dept), x.data(), buffer_r.data(), alpha_cd, dept);

        #pragma omp parallel for
        for (size_type device = 0; device < count_devices; ++device) {
            gpuErrchk(cudaSetDevice(device));
            gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
        }
        if (run % 50 == 49) {
            std::vector<real_type> buffer(b);
            buffer.resize(dept_all);
            gpuErrchk(cudaSetDevice(0));
            gpuErrchk(cudaMemcpy(r_d[0], buffer.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
            #pragma omp parallel for
            for (size_type device = 1; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(cudaMemset(r_d[device], 0, dept_all * sizeof(real_type)));
            }
            switch (kernel_) {
                case kernel_type::linear: {
                    #pragma omp parallel for
                    for (size_type device = 0; device < count_devices; ++device) {
                        gpuErrchk(cudaSetDevice(device));
                        const int start = device * Ncols / count_devices;
                        const int end = (device + 1) * Ncols / count_devices;
                        dim3 grid(static_cast<size_type>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                                  static_cast<size_type>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
                        kernel_linear<<<grid, block>>>(q_d[device], r_d[device], x_d[device], data_d[device], QA_cost_, 1 / cost_, Ncols, Nrows, -1, start, end);
                        gpuErrchk(cudaPeekAtLastError());
                    }
                } break;
                case kernel_type::polynomial:
                    // kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost_, 1/cost_, num_features_, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma_, coef0_, degree_);
                    break;
                case kernel_type::rbf:
                    // kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost_, 1/cost_, num_features_, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma_);
                    break;
                default:
                    throw std::runtime_error("Can not decide wich kernel!");
            }
            gpuErrchk(cudaDeviceSynchronize());
            // cudaMemcpy(r, r_d, dept*sizeof(real_type), cudaMemcpyDeviceToHost);

            {
                gpuErrchk(cudaSetDevice(0));
                gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
                #pragma omp parallel for
                for (size_type device = 1; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    std::vector<real_type> ret(dept_all, 0);
                    gpuErrchk(
                        cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
                    for (size_type j = 0; j <= dept; ++j) {
                        r[j] += ret[j];
                    }
                }
                #pragma omp parallel for
                for (size_type device = 0; device < count_devices; ++device) {
                    gpuErrchk(cudaSetDevice(device));
                    gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
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
        beta = -mult(r.data(), Ad.data(), dept) / mult(d, Ad.data(), dept);  // TODO:
        add(mult(beta, d, dept), r.data(), d, dept);                         // TODO:

        {
            std::vector<real_type> buffer(dept_all, 0.0);
            std::copy(d, d + dept, buffer.begin());
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                gpuErrchk(cudaSetDevice(device));
                gpuErrchk(
                    cudaMemcpy(r_d[device], buffer.data(), dept_all * sizeof(real_type), cudaMemcpyHostToDevice));
            }
        }
    }
    if (run == imax) {
        std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
    }

    alpha_.resize(dept);
    std::vector<real_type> ret_q(dept);
    gpuErrchk(cudaDeviceSynchronize());
    {
        std::vector<real_type> buffer(dept_all);
        std::copy(x.begin(), x.begin() + dept, alpha_.begin());
        gpuErrchk(cudaSetDevice(0));
        gpuErrchk(cudaMemcpy(buffer.data(), q_d[0], dept_all * sizeof(real_type), cudaMemcpyDeviceToHost));
        std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
    }
    // cudaMemcpy(&alpha[0],x_d, dept * sizeof(real_type), cudaMemcpyDeviceToHost);
    // cudaMemcpy(&ret_q[0],q_d, dept * sizeof(real_type), cudaMemcpyDeviceToHost);
    // cudaFree(Ad_d);
    // cudaFree(r_d);
    // cudaFree(datlast);
    // cudaFreeHost(Ad);
    // cudaFree(x_d);
    // cudaFreeHost(r);
    // cudaFreeHost(d);
    return alpha_;
}

template class CUDA_CSVM<float>;
template class CUDA_CSVM<double>;

}  // namespace plssvm
