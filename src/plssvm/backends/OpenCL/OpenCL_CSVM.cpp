#include <plssvm/backends/CUDA/cuda-kernel.hpp>
#include <plssvm/backends/OpenCL/OpenCL_CSVM.hpp>
#include <plssvm/detail/operators.hpp>

#include "manager/configuration.hpp"
#include "manager/device.hpp"
#include "manager/manager.hpp"
#include <plssvm/backends/OpenCL/DevicePtrOpenCL.hpp>
#include <stdexcept>

#include "manager/apply_arguments.hpp"
#include "manager/run_kernel.hpp"

#include <chrono>

namespace plssvm {

// TODO:
std::size_t count_devices = 1;

template <typename T>
OpenCL_CSVM<T>::OpenCL_CSVM(parameter<T> &params) :
    OpenCL_CSVM{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
OpenCL_CSVM<T>::OpenCL_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
    CSVM<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    std::vector<opencl::device_t> &devices = manager.get_devices();
    first_device = devices[0];
    count_devices = devices.size();
    svm_kernel_linear.resize(count_devices, nullptr);
    kernel_q_cl.resize(count_devices, nullptr);
    std::cout << "GPUs found: " << count_devices << std::endl;
}

template <typename T>
void OpenCL_CSVM<T>::setup_data_on_device() {
    std::vector<opencl::device_t> &devices = manager.get_devices();  //TODO: header
    for (size_type device = 0; device < count_devices; ++device) {
        datlast_cl.emplace_back(opencl::DevicePtrOpenCL<real_type>(devices[device], (num_features_)));
    }

    std::vector<real_type> datalast(data_[num_data_points_ - 1]);
    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        datlast_cl[device].to_device(datalast);
    }

    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        datlast_cl[device].resize(num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
    }

    for (size_type device = 0; device < count_devices; ++device) {
        data_cl.emplace_back(
            opencl::DevicePtrOpenCL<real_type>(devices[device], num_features_ * (num_data_points_ - 1)));
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
        data_cl[device] = opencl::DevicePtrOpenCL<real_type>(devices[device], num_features_ * (num_data_points_ - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE));
        data_cl[device].to_device(transformet_data);
    }
}

template <typename T>
void OpenCL_CSVM<T>::resizeData(const int device, const int boundary) {
    std::vector<opencl::device_t> &devices = manager.get_devices();  //TODO: header

    data_cl[device] = opencl::DevicePtrOpenCL<real_type>(devices[device], num_features_ * (num_data_points_ - 1 + boundary));
    std::vector<real_type> vec;
    //vec.reserve(num_data_points + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1);
    for (size_type col = 0; col < num_features_; ++col) {
        for (size_type row = 0; row < num_data_points_ - 1; ++row) {
            vec.push_back(data_[row][col]);
        }
        for (size_type i = 0; i < boundary; ++i) {
            vec.push_back(0.0);
        }
    }
    data_cl[device].to_device(vec);
}

template <typename T>
auto OpenCL_CSVM<T>::generate_q() -> std::vector<real_type> {
    std::vector<opencl::device_t> &devices = manager.get_devices();  //TODO: header
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;

    std::vector<opencl::DevicePtrOpenCL<real_type>> q_cl;
    for (size_type device = 0; device < count_devices; ++device) {
        q_cl.emplace_back(devices[device], dept_all);
    }
    //TODO: init on gpu
    for (size_type device = 0; device < count_devices; ++device) {
        q_cl[device].to_device(std::vector<real_type>(dept_all, 0.0));
    }
    std::cout << "kernel_q" << std::endl;
    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        if (!kernel_q_cl[device]) {
            #pragma omp critical  //TODO: evtl besser keine Referenz
            {
                std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/kernel_q.cl" };
                std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                if (*typeid(real_t).name() == 'f') {
                    manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                } else if (*typeid(real_t).name() == 'd') {
                    manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                }
                json::node &deviceNode =
                    manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                               ["DEVICES"][devices[device].deviceName];
                json::node &kernelConfig = deviceNode["KERNELS"]["kernel_q"];
                kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
                kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
                kernel_q_cl[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_q");
            }
        }
        const int Ncols = num_features_;
        const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;

        q_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
        // resizeData(i,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
        const int start = device * Ncols / count_devices;
        const int end = (device + 1) * Ncols / count_devices;
        opencl::apply_arguments(kernel_q_cl[device], q_cl[device].get(), data_cl[device].get(), datlast_cl[device].get(), Nrows, start, end);

        size_type grid_size = static_cast<size_t>(
            ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE)) * THREADBLOCK_SIZE);
        size_type block_size = THREADBLOCK_SIZE;
        opencl::run_kernel_1d_timed(devices[device], kernel_q_cl[device], grid_size, block_size);
    }

    std::vector<real_type> q(dept_all);
    q_cl[0].from_device(q);  //TODO:
    std::vector<real_type> ret(dept);
    for (size_type device = 1; device < count_devices; ++device) {
        q_cl[device].from_device(ret);
        for (size_type i = 0; i < dept; ++i) {
            q[i] += ret[i];
        }
    }
    q.resize(dept);
    return q;
}

template <typename T>
auto OpenCL_CSVM<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    std::vector<opencl::device_t> &devices = manager.get_devices();  //TODO: header
    const size_type dept = num_data_points_ - 1;
    const size_type boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
    const size_type dept_all = dept + boundary_size;
    std::vector<real_type> zeros(dept_all, 0.0);

    real_type *d;
    std::vector<real_type> x(dept_all, 1.0);
    std::fill(x.end() - boundary_size, x.end(), 0.0);

    std::vector<opencl::DevicePtrOpenCL<real_type>> x_cl;
    for (size_type device = 0; device < count_devices; ++device) {
        x_cl.emplace_back(devices[device], dept_all);
    }
    for (size_type device = 0; device < count_devices; ++device) {
        x_cl[device].to_device(x);
    }

    std::vector<real_type> r(dept_all, 0.0);

    std::vector<opencl::DevicePtrOpenCL<real_type>> r_cl;
    for (size_type device = 0; device < count_devices; ++device)
        r_cl.emplace_back(devices[device], dept_all);

    {
        std::vector<real_type> toDevice(dept_all, 0.0);
        std::copy(b.begin(), b.begin() + dept, toDevice.begin());
        r_cl[0].to_device(std::vector<real_type>(toDevice));
    }
    #pragma omp parallel for
    for (size_type device = 1; device < count_devices; ++device) {
        r_cl[device].to_device(std::vector<real_type>(zeros));
    }
    d = new real_type[dept];

    std::vector<opencl::DevicePtrOpenCL<real_type>> q_cl;
    for (size_type device = 0; device < count_devices; ++device) {
        q_cl.emplace_back(devices[device], q.size());
    }

    #pragma omp parallel
    for (size_type device = 0; device < count_devices; ++device) {
        q_cl[device].to_device(q);
    }

    for (size_type device = 0; device < count_devices; ++device) {
        q_cl[device].resize(dept_all);
    }

    switch (kernel_) {
        case kernel_type::linear:
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                if (!svm_kernel_linear[device]) {
                    std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel-linear.cl" };
                    std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                    if (*typeid(real_t).name() == 'f') {
                        manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                    } else if (*typeid(real_t).name() == 'd') {
                        manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                    }
                    json::node &deviceNode =
                        manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                   ["DEVICES"][devices[device].deviceName];
                    json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                    #pragma omp critical  //TODO: evtl besser keine Referenz
                    {
                        kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
                        kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
                        svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                    }
                }
                {
                    q_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                    r_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                    x_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                    // resizeData(device,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                    const int Ncols = num_features_;
                    const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                    std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                                                   static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))) };
                    const int start = device * Ncols / count_devices;
                    const int end = (device + 1) * Ncols / count_devices;
                    opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), r_cl[device].get(), x_cl[device].get(), data_cl[device].get(), QA_cost_, 1 / cost_, Ncols, Nrows, -1, start, end);
                    grid_size[0] *= THREADBLOCK_SIZE;
                    grid_size[1] *= THREADBLOCK_SIZE;
                    std::vector<size_type> block_size{ THREADBLOCK_SIZE, THREADBLOCK_SIZE };

                    opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
                    // exit(0);
                }
            }
            break;

        case kernel_type::polynomial:
            //TODO: kernel_poly OpenCL
            std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
            break;
        case kernel_type::rbf:
            //TODO: kernel_radial OpenCL
            std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
            break;
        default:
            throw std::runtime_error("Can not decide wich kernel!");
    }

    {
        r_cl[0].resize(dept_all);
        r_cl[0].from_device(r);
        for (size_type device = 1; device < count_devices; ++device) {
            std::vector<real_type> ret(dept_all);
            r_cl[device].from_device(ret);
            for (size_type j = 0; j <= dept; ++j) {
                r[j] += ret[j];
            }
        }

        for (size_type device = 0; device < count_devices; ++device)
            r_cl[device].to_device(r);
    }
    real_type delta = mult(r.data(), r.data(), dept);  //TODO:
    const real_type delta0 = delta;
    real_type alpha_cd, beta;
    std::vector<real_type> Ad(dept);

    std::vector<opencl::DevicePtrOpenCL<real_type>> Ad_cl;
    for (size_type device = 0; device < count_devices; ++device) {
        Ad_cl.emplace_back(devices[device], dept_all);
    }

    size_type run;
    for (run = 0; run < imax; ++run) {
        if (print_info_) {
            std::cout << "Start Iteration: " << run << std::endl;
        }
        //Ad = A * d
        {
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                Ad_cl[device].to_device(zeros);
            }
            //TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                std::vector<real_type> buffer(dept_all);
                r_cl[device].resize(dept_all);
                r_cl[device].from_device(buffer);
                for (size_type index = dept; index < dept_all; ++index) {
                    buffer[index] = 0.0;
                }
                r_cl[device].to_device(buffer);
            }
        }
        switch (kernel_) {
            case kernel_type::linear:
                #pragma omp parallel for
                for (size_type device = 0; device < count_devices; ++device) {
                    if (!svm_kernel_linear[device]) {
                        std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel-linear.cl" };
                        std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                        if (*typeid(real_t).name() == 'f') {
                            manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                        } else if (*typeid(real_t).name() == 'd') {
                            manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                        }
                        json::node &deviceNode =
                            manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                       ["DEVICES"][devices[device].deviceName];
                        json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                        #pragma omp critical
                        {
                            kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
                            kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
                            svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                        }
                    }
                    {
                        q_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                        Ad_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                        r_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                        // resizeData(device,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                        const int Ncols = num_features_;
                        const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                        std::vector<size_type> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                                                          static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))) };
                        const int start = device * Ncols / count_devices;
                        const int end = (device + 1) * Ncols / count_devices;
                        opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), Ad_cl[device].get(), r_cl[device].get(), data_cl[device].get(), QA_cost_, 1 / cost_, Ncols, Nrows, 1, start, end);
                        grid_size[0] *= THREADBLOCK_SIZE;
                        grid_size[1] *= THREADBLOCK_SIZE;
                        std::vector<size_type> block_size{ THREADBLOCK_SIZE, THREADBLOCK_SIZE };

                        opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
                    }
                }
                break;
            case kernel_type::polynomial:
                //TODO: kernel_poly OpenCL
                std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
                break;
            case kernel_type::rbf:
                //TODO: kernel_radial OpenCL
                std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
                break;
            default:
                throw std::runtime_error("Can not decide which kernel!");
        }

        for (size_type i = 0; i < dept; ++i)
            d[i] = r[i];

        {
            std::vector<real_type> buffer(dept_all, 0);
            for (size_type device = 0; device < count_devices; ++device) {
                // for(int device = 0; device < 1; ++device){
                std::vector<real_type> ret(dept_all, 0);
                Ad_cl[device].resize(dept_all);
                Ad_cl[device].from_device(ret);
                for (size_type j = 0; j < dept; ++j) {
                    buffer[j] += ret[j];
                }
            }
            std::copy(buffer.begin(), buffer.begin() + dept, Ad.data());
            for (size_type device = 0; device < count_devices; ++device)
                Ad_cl[device].to_device(buffer);
        }

        alpha_cd = delta / mult(d, Ad.data(), dept);
        //TODO: auf GPU
        std::vector<real_type> buffer_r(dept_all);
        r_cl[0].resize(dept_all);
        r_cl[0].from_device(buffer_r);
        add_mult_(((int) dept / 1024) + 1, std::min(1024, (int) dept), x.data(), buffer_r.data(), alpha_cd, dept);
        #pragma omp parallel
        for (size_type device = 0; device < count_devices; ++device) {
            x_cl[device].resize(dept_all);
        }
        #pragma omp parallel
        for (size_type device = 0; device < count_devices; ++device) {
            x_cl[device].to_device(x);
        }

        if (run % 50 == 49) {
            std::vector<real_type> buffer(b);
            buffer.resize(dept_all);
            r_cl.resize(dept_all);
            r_cl[0].to_device(buffer);
            #pragma omp parallel for
            for (size_type device = 1; device < count_devices; ++device) {
                r_cl[device].to_device(zeros);
            }
            switch (kernel_) {
                case kernel_type::linear:
                    #pragma omp parallel for
                    for (size_type device = 0; device < count_devices; ++device) {
                        if (!svm_kernel_linear[device]) {
                            std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel-linear.cl" };
                            std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                            if (*typeid(real_t).name() == 'f') {
                                manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                            } else if (*typeid(real_t).name() == 'd') {
                                manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                            }
                            json::node &deviceNode =
                                manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                           ["DEVICES"][devices[device].deviceName];
                            json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                            #pragma omp critical
                            {
                                kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE",
                                                             std::to_string(INTERNALBLOCK_SIZE));
                                kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
                                svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                            }
                        }

                        {
                            q_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                            r_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                            x_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                            // resizeData(device,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
                            const int Ncols = num_features_;
                            const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
                            const int start = device * Ncols / count_devices;
                            const int end = (device + 1) * Ncols / count_devices;
                            std::vector<size_type> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),
                                                              static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))) };
                            opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), r_cl[device].get(), x_cl[device].get(), data_cl[device].get(), QA_cost_, 1 / cost_, Ncols, Nrows, -1, start, end);
                            grid_size[0] *= THREADBLOCK_SIZE;
                            grid_size[1] *= THREADBLOCK_SIZE;
                            std::vector<size_type> block_size{ THREADBLOCK_SIZE, THREADBLOCK_SIZE };

                            opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
                        }
                    }
                    break;
                case kernel_type::polynomial:
                    //TODO: kernel_poly OpenCL
                    std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
                    break;
                case kernel_type::rbf:
                    //TODO: kernel_radial OpenCL
                    std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
                    break;
                default:
                    throw std::runtime_error("Can not decide wich kernel!");
            }

            {
                r_cl[0].resize(dept_all);
                r_cl[0].from_device(r);
                for (size_type device = 1; device < count_devices; ++device) {
                    std::vector<real_type> ret(dept_all, 0);
                    r_cl[device].resize(dept_all);
                    r_cl[device].from_device(ret);
                    for (size_type j = 0; j <= dept; ++j) {
                        r[j] += ret[j];
                    }
                }
                #pragma omp parallel for
                for (size_type device = 0; device < count_devices; ++device) {
                    r_cl[device].to_device(r);
                }
            }
        } else {
            for (size_type index = 0; index < dept; ++index) {
                r[index] -= alpha_cd * Ad[index];
            }
        }

        delta = mult(r.data(), r.data(), dept);  //TODO:
        if (delta < eps * eps * delta0) {
            break;
        }
        beta = -mult(r.data(), Ad.data(), dept) / mult(d, Ad.data(), dept);  //TODO:
        add(mult(beta, d, dept), r.data(), d, dept);                         //TODO:

        {
            std::vector<real_type> buffer(dept_all, 0.0);
            std::copy(d, d + dept, buffer.begin());
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                r_cl[device].resize(dept_all);
            }
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                r_cl[device].to_device(buffer);
            }
        }
    }
    if (run == imax) {
        std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
    }

    alpha_.resize(dept);
    std::vector<real_type> ret_q(dept);

    {
        std::vector<real_type> buffer(dept_all);
        std::copy(x.begin(), x.begin() + dept, alpha_.begin());
        q_cl[0].resize(dept_all);
        q_cl[0].from_device(buffer);
        std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
    }
    delete[] d;
    return alpha_;
}

// explicitly instantiate template class
template class OpenCL_CSVM<float>;
template class OpenCL_CSVM<double>;

}  // namespace plssvm