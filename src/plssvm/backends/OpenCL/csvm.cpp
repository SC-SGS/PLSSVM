/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/OpenCL/csvm.hpp"

#include "plssvm/backends/OpenCL/detail/device_ptr.hpp"  // plssvm::detail::opencl::device_ptr
#include "plssvm/backends/OpenCL/exceptions.hpp"         // plssvm::opencl::backend_exception
#include "plssvm/constants.hpp"                          // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm.hpp"                               // plssvm::csvm
#include "plssvm/detail/arithmetic_type_name.hpp"        // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/operators.hpp"                   // various operator overloads for std::vector and scalars
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::replace_all
#include "plssvm/target_platform.hpp"                    // plssvm::target_platform

#include "manager/apply_arguments.hpp"
#include "manager/configuration.hpp"
#include "manager/device.hpp"
#include "manager/manager.hpp"
#include "manager/run_kernel.hpp"

#include "CL/cl.h"     //
#include "fmt/core.h"  // fmt::print, fmt::format

#include <algorithm>  // std::min
#include <cmath>      // std::ceil
#include <cstdio>     // stderr
#include <exception>  // std::exception, std::terminate
#include <vector>     // std::vector

#include <fstream>
#include <streambuf>
#include <string>

#include <chrono>
#include <stdexcept>

namespace plssvm::opencl {

// TODO:
std::size_t count_devices = 1;

cl_context context;

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    csvm{ params.target, params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
csvm<T>::csvm(const target_platform target, const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    ::plssvm::csvm<T>{ target, kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    if (print_info_) {
        fmt::print("Using OpenCL as backend.\n");
    }

    // TODO: RAII
    // get all available devices wrt the requested target platform
    cl_int err;
    cl_device_id device_id;
    clGetDeviceIDs(nullptr, CL_DEVICE_TYPE_GPU, 1, &device_id, nullptr);
    context = clCreateContext(0, 1, &device_id, nullptr, nullptr, &err);
    devices_.emplace_back(clCreateCommandQueue(context, device_id, 0, &err));

    // throw exception if no devices for the requested target could be found
    if (devices_.size() < 1) {
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
        // TODO:
        //        for (size_type device = 0; device < devices_.size(); ++device) {
        //            fmt::print("  [{}, {}]\n", device, devices_[device].get_device().template get_info<::sycl::info::device::name>());
        //        }
        //        fmt::print("\n");
    }

    std::vector<::opencl::device_t> &devices = manager.get_devices();
    first_device = devices[0];
    count_devices = devices.size();
    svm_kernel_linear.resize(count_devices, nullptr);
    kernel_q_cl.resize(count_devices, nullptr);
    std::cout << "GPUs found: " << count_devices << '\n'
              << std::endl;
}

template <typename T>
csvm<T>::~csvm() {
    // TODO:
}

template <typename T>
void csvm<T>::setup_data_on_device() {
    // set values of member variables
    dept_ = num_data_points_ - 1;
    boundary_size_ = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    num_rows_ = dept_ + boundary_size_;
    num_cols_ = num_features_;

    std::vector<::opencl::device_t> &devices = manager.get_devices();  //TODO: header
    // initialize data_last on device
    for (size_type device = 0; device < devices.size(); ++device) {
        data_last_d_[device] = detail::device_ptr<real_type>{ num_features_ + boundary_size_, devices[device].commandQueue };
    }
    #pragma omp parallel for
    for (size_type device = 0; device < devices.size(); ++device) {
        data_last_d_[device].memset(0);
        data_last_d_[device].memcpy_to_device(data_[dept_], 0, num_features_);
    }

    // initialize data on devices
    for (size_type device = 0; device < devices.size(); ++device) {
        data_d_[device] = detail::device_ptr<real_type>{ num_features_ * (dept_ + boundary_size_), devices[device].commandQueue };
    }
    // transform 2D to 1D data
    const std::vector<real_type> transformed_data = base_type::transform_data(boundary_size_);
    #pragma omp parallel for
    for (size_type device = 0; device < devices.size(); ++device) {
        data_d_[device].memcpy_to_device(transformed_data, 0, num_features_ * (dept_ + boundary_size_));
    }
}

template <typename T>
auto csvm<T>::generate_q() -> std::vector<real_type> {
    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_rows_ != 0, "num_rows_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(num_cols_ != 0, "num_cols_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<::opencl::device_t> &devices = manager.get_devices();

    std::vector<detail::device_ptr<real_type>> q_d(devices.size());
    for (size_type device = 0; device < devices.size(); ++device) {
        q_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices[device].commandQueue };
        q_d[device].memset(0);
    }

    //    for (size_type device = 0; device < devices_.size(); ++device) {
    //        // feature splitting on multiple devices
    //        const int first_feature = device * num_cols_ / devices_.size();
    //        const int last_feature = (device + 1) * num_cols_ / devices_.size();
    //
    //        const auto grid = static_cast<size_type>(std::ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE)));
    //        const size_type block = std::min<size_type>(THREAD_BLOCK_SIZE, dept_);
    //
    //        // TODO: add switch
    //        // TODO: call kernel
    //        std::ifstream in("../src/plssvm/backends/OpenCL/kernels/kernel_q.cl");
    //        std::string kernel_src((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    //
    //        // replace type
    //        ::plssvm::detail::replace_all(kernel_src, "real_type", ::plssvm::detail::arithmetic_type_name<real_type>());
    //        // replace constants
    //        ::plssvm::detail::replace_all(kernel_src, "INTERNAL_BLOCK_SIZE", std::to_string(INTERNAL_BLOCK_SIZE));
    //        ::plssvm::detail::replace_all(kernel_src, "THREAD_BLOCK_SIZE", std::to_string(THREAD_BLOCK_SIZE));
    //
    //        cl_int err;
    //        const char *kernel_src_ptr = kernel_src.c_str();
    //        cl_program program = clCreateProgramWithSource(context, 1, &kernel_src_ptr, nullptr, &err);
    //        fmt::print("{}: {}\n", err, "clCreateProgramWithSource");
    //        err = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    //        fmt::print("{}: {}\n", err, "clBuildProgram");
    //        cl_kernel kernel = clCreateKernel(program, "kernel_q", &err);
    //        fmt::print("{}: {}\n", err, "clCreateKernel");
    //        cl_mem ptr = q_d[device].get();
    //        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &ptr);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 0");
    //        ptr = data_d_[device].get();
    //        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &ptr);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 1");
    //        ptr = data_last_d_[device].get();
    //        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &ptr);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 2");
    //        err = clSetKernelArg(kernel, 3, sizeof(int), &num_rows_);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 3");
    //        err = clSetKernelArg(kernel, 4, sizeof(int), &first_feature);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 4");
    //        err = clSetKernelArg(kernel, 5, sizeof(int), &last_feature);
    //        fmt::print("{}: {}\n", err, "clSetKernelArg 5");
    //        err = clEnqueueNDRangeKernel(devices_[device], kernel, 1, nullptr, &grid, &block, 0, nullptr, nullptr);
    //        fmt::print("{}: {}\n", err, "clEnqueueNDRangeKernel");
    //        err = clFinish(devices_[device]);
    //        fmt::print("{}: {}\n", err, "clFinish");
    //    }
    //
    //    std::vector<real_type> q(dept_);
    //    device_reduction(q_d, q);
    //    return q;

    //    std::vector<::opencl::device_t> &devices = manager.get_devices();  //TODO: header
    //    const size_type dept = num_data_points_ - 1;
    //    const size_type boundary_size = THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE;
    //    const size_type dept_all = dept + boundary_size;
    //
    //    std::vector<::opencl::DevicePtrOpenCL<real_type>> q_cl;
    //    for (size_type device = 0; device < count_devices; ++device) {
    //        q_cl.emplace_back(devices[device], dept_all);
    //    }
    //    //TODO: init on gpu
    //    for (size_type device = 0; device < count_devices; ++device) {
    //        q_cl[device].to_device(std::vector<real_type>(dept_all, 0.0));
    //    }
    //    std::cout << "kernel_q" << std::endl;
    #pragma omp parallel for
    for (size_type device = 0; device < count_devices; ++device) {
        if (!kernel_q_cl[device]) {
            #pragma omp critical  //TODO: evtl besser keine Referenz
            {
                std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/kernel_q.cl" };
                std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                if constexpr (std::is_same_v<real_type, float>) {
                    ::plssvm::detail::replace_all(kernel_src, "real_type", "float");
                    manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                } else if constexpr (std::is_same_v<real_type, double>) {
                    ::plssvm::detail::replace_all(kernel_src, "real_type", "double");
                    manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                }
                json::node &deviceNode =
                    manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                               ["DEVICES"][devices[device].deviceName];
                json::node &kernelConfig = deviceNode["KERNELS"]["kernel_q"];
                kernelConfig.replaceTextAttr("INTERNAL_BLOCK_SIZE", std::to_string(INTERNAL_BLOCK_SIZE));
                kernelConfig.replaceTextAttr("THREAD_BLOCK_SIZE", std::to_string(THREAD_BLOCK_SIZE));
                kernel_q_cl[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_q");
            }
        }
        // resizeData(i,THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
        const int start = device * num_cols_ / count_devices;
        const int end = (device + 1) * num_cols_ / count_devices;
        ::opencl::apply_arguments(kernel_q_cl[device], q_d[device].get(), data_d_[device].get(), data_last_d_[device].get(), num_rows_, start, end);

        size_type grid_size = static_cast<size_type>(
            ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE)) * THREAD_BLOCK_SIZE);
        size_type block_size = THREAD_BLOCK_SIZE;
        ::opencl::run_kernel_1d_timed(devices[device], kernel_q_cl[device], grid_size, block_size);
    }

    std::vector<real_type> q(dept_);
    device_reduction(q_d, q);
    return q;
}

template <typename T>
void csvm<T>::device_reduction(std::vector<detail::device_ptr<real_type>> &buffer_d, std::vector<real_type> &buffer) {
    std::vector<::opencl::device_t> &devices = manager.get_devices();
    clFinish(devices[0].commandQueue);
    buffer_d[0].memcpy_to_host(buffer, 0, buffer.size());

    if (devices.size() > 1) {
        std::vector<real_type> ret(buffer.size());
        for (size_type device = 1; device < devices.size(); ++device) {
            clFinish(devices[device].commandQueue);
            buffer_d[device].memcpy_to_host(ret, 0, ret.size());

            #pragma omp parallel for
            for (size_type j = 0; j < ret.size(); ++j) {
                buffer[j] += ret[j];
            }
        }

        #pragma omp parallel for
        for (size_type device = 0; device < devices.size(); ++device) {
            buffer_d[device].memcpy_to_device(buffer, 0, buffer.size());
        }
    }
}

template <typename T>
auto csvm<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    using namespace plssvm::operators;

    PLSSVM_ASSERT(dept_ != 0, "dept_ not initialized! Maybe a call to setup_data_on_device() is missing?");
    PLSSVM_ASSERT(boundary_size_ != 0, "boundary_size_ not initialized! Maybe a call to setup_data_on_device() is missing?");

    std::vector<::opencl::device_t> &devices = manager.get_devices();

    std::vector<real_type> x(dept_, 1.0);
    std::vector<detail::device_ptr<real_type>> x_d(devices.size());

    std::vector<real_type> r(dept_, 0.0);
    std::vector<detail::device_ptr<real_type>> r_d(devices.size());

    for (size_type device = 0; device < devices.size(); ++device) {
        x_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices[device].commandQueue };
        r_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices[device].commandQueue };
    }
    #pragma omp parallel for
    for (size_type device = 0; device < devices.size(); ++device) {
        x_d[device].memset(0);
        x_d[device].memcpy_to_device(x, 0, dept_);
        r_d[device].memset(0);
    }
    r_d[0].memcpy_to_device(b, 0, dept_);

    std::vector<detail::device_ptr<real_type>> q_d(devices.size());
    for (size_type device = 0; device < devices.size(); ++device) {
        q_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices[device].commandQueue };
    }
    #pragma omp parallel for
    for (size_type device = 0; device < devices.size(); ++device) {
        q_d[device].memset(0);
        q_d[device].memcpy_to_device(q, 0, dept_);
    }
    switch (kernel_) {
        case kernel_type::linear:
            #pragma omp parallel for
            for (size_type device = 0; device < count_devices; ++device) {
                if (!svm_kernel_linear[device]) {
                    std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel.cl" };
                    std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                    if constexpr (std::is_same_v<real_type, float>) {
                        manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                        ::plssvm::detail::replace_all(kernel_src, "real_type", "float");
                    } else if constexpr (std::is_same_v<real_type, double>) {
                        manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                        ::plssvm::detail::replace_all(kernel_src, "real_type", "double");
                    }
                    json::node &deviceNode =
                        manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                   ["DEVICES"][devices[device].deviceName];
                    json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                    #pragma omp critical  //TODO: evtl besser keine Referenz
                    {
                        kernelConfig.replaceTextAttr("INTERNAL_BLOCK_SIZE", std::to_string(INTERNAL_BLOCK_SIZE));
                        kernelConfig.replaceTextAttr("THREAD_BLOCK_SIZE", std::to_string(THREAD_BLOCK_SIZE));
                        svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                    }
                }
                {
                    std::vector<size_type> grid_size{ static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))),
                                                      static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))) };
                    const int start = device * num_cols_ / count_devices;
                    const int end = (device + 1) * num_cols_ / count_devices;
                    std::flush(std::cout);
                    ::opencl::apply_arguments(svm_kernel_linear[device], q_d[device].get(), r_d[device].get(), x_d[device].get(), data_d_[device].get(), QA_cost_, real_type{ 1. } / cost_, num_cols_, num_rows_, -1, start, end);
                    grid_size[0] *= THREAD_BLOCK_SIZE;
                    grid_size[1] *= THREAD_BLOCK_SIZE;
                    std::vector<size_type> block_size{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

                    ::opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
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

    device_reduction(r_d, r);

    // delta = r.T * r
    real_type delta = transposed{ r } * r;
    const real_type delta0 = delta;
    std::vector<real_type> Ad(dept_);

    std::vector<detail::device_ptr<real_type>> Ad_d(devices.size());
    for (size_type device = 0; device < devices.size(); ++device) {
        Ad_d[device] = detail::device_ptr<real_type>{ dept_ + boundary_size_, devices[device].commandQueue };
    }

    std::vector<real_type> d(r);

    size_type run = 0;
    for (; run < imax; ++run) {
        if (print_info_) {
            fmt::print("Start Iteration {} (max: {}) with current residuum {} (target: {}).\n", run + 1, imax, delta, eps * eps * delta0);
        }
        // Ad = A * r (q = A * d)
        #pragma omp parallel for
        for (size_type device = 0; device < devices.size(); ++device) {
            Ad_d[device].memset(0);
            r_d[device].memset(0, dept_);
        }

        switch (kernel_) {
            case kernel_type::linear:
                #pragma omp parallel for
                for (size_type device = 0; device < count_devices; ++device) {
                    if (!svm_kernel_linear[device]) {
                        std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel.cl" };
                        std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                        if constexpr (std::is_same_v<real_type, float>) {
                            manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                            ::plssvm::detail::replace_all(kernel_src, "real_type", "float");
                        } else if constexpr (std::is_same_v<real_type, double>) {
                            manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                            ::plssvm::detail::replace_all(kernel_src, "real_type", "double");
                        }
                        json::node &deviceNode =
                            manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                       ["DEVICES"][devices[device].deviceName];
                        json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                        #pragma omp critical
                        {
                            kernelConfig.replaceTextAttr("INTERNAL_BLOCK_SIZE", std::to_string(INTERNAL_BLOCK_SIZE));
                            kernelConfig.replaceTextAttr("THREAD_BLOCK_SIZE", std::to_string(THREAD_BLOCK_SIZE));
                            svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                        }
                    }
                    {
                        // resizeData(device,THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE);
                        std::vector<size_type> grid_size{ static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))),
                                                          static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))) };
                        const int start = device * num_cols_ / count_devices;
                        const int end = (device + 1) * num_cols_ / count_devices;
                        ::opencl::apply_arguments(svm_kernel_linear[device], q_d[device].get(), Ad_d[device].get(), r_d[device].get(), data_d_[device].get(), QA_cost_, real_type{ 1. } / cost_, num_cols_, num_rows_, 1, start, end);
                        grid_size[0] *= THREAD_BLOCK_SIZE;
                        grid_size[1] *= THREAD_BLOCK_SIZE;
                        std::vector<size_type> block_size{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

                        ::opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
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

        // update Ad (q)
        device_reduction(Ad_d, Ad);

        // (alpha = delta_new / (d^T * q))
        const real_type alpha_cd = delta / (transposed{ d } * Ad);

        // (x = x + alpha * d)
        x += alpha_cd * d;

        #pragma omp parallel for
        for (size_type device = 0; device < devices.size(); ++device) {
            x_d[device].memcpy_to_device(x, 0, dept_);
        }

        if (run % 50 == 49) {
            // r = b
            r_d[0].memcpy_to_device(b, 0, dept_);
            #pragma omp parallel for
            for (size_type device = 1; device < devices.size(); ++device) {
                r_d[device].memset(0);
            }

            // r -= A * x
            switch (kernel_) {
                case kernel_type::linear:
                    #pragma omp parallel for
                    for (size_type device = 0; device < count_devices; ++device) {
                        if (!svm_kernel_linear[device]) {
                            std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel.cl" };
                            std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                            if constexpr (std::is_same_v<real_type, float>) {
                                manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
                                ::plssvm::detail::replace_all(kernel_src, "real_type", "float");
                            } else if constexpr (std::is_same_v<real_type, double>) {
                                manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
                                ::plssvm::detail::replace_all(kernel_src, "real_type", "double");
                            }
                            json::node &deviceNode =
                                manager.get_configuration()["PLATFORMS"][devices[device].platformName]
                                                           ["DEVICES"][devices[device].deviceName];
                            json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                            #pragma omp critical
                            {
                                kernelConfig.replaceTextAttr("INTERNAL_BLOCK_SIZE",
                                                             std::to_string(INTERNAL_BLOCK_SIZE));
                                kernelConfig.replaceTextAttr("THREAD_BLOCK_SIZE", std::to_string(THREAD_BLOCK_SIZE));
                                svm_kernel_linear[device] = manager.build_kernel(kernel_src, devices[device], kernelConfig, "kernel_linear");
                            }
                        }

                        {
                            const int start = device * num_cols_ / count_devices;
                            const int end = (device + 1) * num_cols_ / count_devices;
                            std::vector<size_type> grid_size{ static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))),
                                                              static_cast<size_type>(ceil(static_cast<real_type>(dept_) / static_cast<real_type>(THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE))) };
                            ::opencl::apply_arguments(svm_kernel_linear[device], q_d[device].get(), r_d[device].get(), x_d[device].get(), data_d_[device].get(), QA_cost_, real_type{ 1. } / cost_, num_cols_, num_rows_, -1, start, end);
                            grid_size[0] *= THREAD_BLOCK_SIZE;
                            grid_size[1] *= THREAD_BLOCK_SIZE;
                            std::vector<size_type> block_size{ THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE };

                            ::opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
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
        for (size_type device = 0; device < devices.size(); ++device) {
            r_d[device].memcpy_to_device(d, 0, dept_);
        }
    }
    if (print_info_) {
        fmt::print("Finished after {} iterations with a residuum of {} (target: {}).\n", run + 1, delta, eps * eps * delta0);
    }

    alpha_.assign(x.begin(), x.begin() + dept_);
    // alpha_.resize(dept);
    // x_d[0].memcpy_to_host(alpha_, 0, dept);

    return alpha_;
}

// explicitly instantiate template class
template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::opencl
