#include <plssvm/CSVM.hpp>
#include <plssvm/CUDA/cuda-kernel.cuh>
#include <plssvm/CUDA/cuda-kernel.hpp>
#include <plssvm/CUDA/svm-kernel.cuh>

#ifdef PLSSVM_HAS_OPENCL_BACKEND
#include "manager/configuration.hpp"
#include "manager/device.hpp"
#include "manager/manager.hpp"
#include <plssvm/OpenCL/DevicePtrOpenCL.hpp>
#include <stdexcept>

#include "manager/apply_arguments.hpp"
#include "manager/run_kernel.hpp"
#endif

using namespace opencl;
int CUDADEVICE = 0;

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_, bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_), kernel_q_cl(nullptr), svm_kernel_linear(nullptr) {

#ifdef PLSSVM_HAS_OPENCL_BACKEND

    std::vector<opencl::device_t> &devices = manager.get_devices();

    /*if (devices.size() > 1) {
			throw std::runtime_error("can only use a single device for OpenCL version");
		}*/
    first_device = devices[0];

#endif
}

void CSVM::learn() {
    std::vector<double> q;
    std::vector<double> b = value;
#pragma omp parallel sections
    {
#pragma omp section // generate right side from eguation
        {
            b.pop_back();
            b -= value.back();
        }
#pragma omp section // generate botom right from A
        {
            QA_cost = kernel_function(data.back(), data.back()) + 1 / cost;
        }
    }

    if (info)
        std::cout << "start CG" << std::endl;
    //solve minimization
    q = CG(b, num_features, epsilon);
    alpha.emplace_back(-sum(alpha));
    bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}

double CSVM::kernel_function(std::vector<double> &xi, std::vector<double> &xj) {
    switch (kernel) {
    case 0:
        return xi * xj;
    case 1:
        return std::pow(gamma * (xi * xj) + coef0, degree);
    case 2: {
        double temp = 0;
        for (int i = 0; i < xi.size(); ++i) {
            temp += (xi - xj) * (xi - xj);
        }
        return exp(-gamma * temp);
    }
    default:
        throw std::runtime_error("Can not decide wich kernel!");
    }
}

void CSVM::loadDataDevice() {

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    datlast_cl = opencl::DevicePtrOpenCL<double>(first_device, (num_features + CUDABLOCK_SIZE - 1));
    std::vector<double> datalast(data[num_data_points - 1]);
    for (int i = 0; i < CUDABLOCK_SIZE - 1; ++i)
        datalast.push_back(0.0);
    datlast_cl.to_device(datalast);

    data_cl = opencl::DevicePtrOpenCL<double>(first_device, num_features * (num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1));
    std::vector<double> vec;
    vec.reserve(num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
    // #pragma parallel for
    for (size_t col = 0; col < num_features; ++col) {
        for (size_t row = 0; row < num_data_points - 1; ++row) {
            vec.push_back(data[row][col]);
        }
        for (int i = 0; i < +(CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD); ++i) {
            vec.push_back(0.0);
        }
    }
    data_cl.to_device(vec);

#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMallocManaged((void **)&datlast, (num_features + CUDABLOCK_SIZE - 1) * sizeof(double));
    cudaMemset(datlast, 0, num_features + CUDABLOCK_SIZE - 1 * sizeof(double));
    cudaMemcpy(datlast, &data[num_data_points - 1][0], num_features * sizeof(double), cudaMemcpyHostToDevice);
    cudaMallocManaged((void **)&data_d, num_features * (num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));

    double *col_vec;
    cudaMallocHost((void **)&col_vec, (num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));

    // #pragma parallel for
    for (size_t col = 0; col < num_features; ++col) {
        for (size_t row = 0; row < num_data_points - 1; ++row) {
            col_vec[row] = data[row][col];
        }
        for (int i = 0; i < +(CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD); ++i) {
            col_vec[i + num_data_points - 1] = 0;
        }
        cudaMemcpy(data_d + col * (num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1), col_vec, (num_data_points + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaFreeHost(col_vec);
#else

#endif
}

void CSVM::learn(std::string &filename, std::string &output_filename) {
    auto begin_parse = std::chrono::high_resolution_clock::now();
    if (filename.size() > 5 && endsWith(filename, ".arff")) {
        arffParser(filename);
    } else {
        libsvmParser(filename);
    }

    auto end_parse = std::chrono::high_resolution_clock::now();
    if (info) {
        std::clog << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << " ms eingelesen" << std::endl
                  << std::endl;
    }

#ifdef PLSSVM_HAS_CUDA_BACKEND
    cudaSetDevice(CUDADEVICE);
#endif
    loadDataDevice();

    auto end_gpu = std::chrono::high_resolution_clock::now();

    if (info)
        std::clog << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << " ms auf die Gpu geladen" << std::endl
                  << std::endl;

    learn();
    auto end_learn = std::chrono::high_resolution_clock::now();
    if (info)
        std::clog << std::endl
                  << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << " ms gelernt" << std::endl;

    writeModel(output_filename);
    auto end_write = std::chrono::high_resolution_clock::now();
    if (info) {
        std::clog << std::endl
                  << data.size() << " Datenpunkte mit Dimension " << num_features << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << " geschrieben" << std::endl;
    } else if (times) {
        std::clog << data.size() << ", " << num_features << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_write - end_learn).count() << std::endl;
    }
}

std::vector<double> CSVM::CG(const std::vector<double> &b, const int imax, const double eps) {
    const int dept = num_data_points - 1;

#ifdef PLSSVM_HAS_OPENCL_BACKEND
//TODO
#elif PLSSVM_HAS_CUDA_BACKEND
    dim3 grid((int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1, (int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1);
    dim3 block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
    double *x_d, *r_d, *q_d;
#else
//TODO
#endif

    double *x, *r, *d;

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    opencl::DevicePtrOpenCL<double> x_cl(first_device, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMalloc((void **)&x_d, (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
#else

#endif
    x = new double[(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1)];

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    //TODO
    init_(((int)dept / 1024) + 1, std::min(1024, dept), x, 1, dept);
    init_(1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1, x + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
    x_cl.to_device(std::vector<double>(x, x + (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1)));
#elif PLSSVM_HAS_CUDA_BACKEND
    init<<<((int)dept / 1024) + 1, std::min(1024, dept)>>>(x_d, 1, dept);
    init<<<1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1>>>(x_d + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#else
    init_(((int)dept / 1024) + 1, std::min(1024, dept), x, 1, dept);
    init_(1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1, x + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#endif

    cudaDeviceSynchronize();

    r = new double[(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1)];

//r = b - (A * x)
///r = b;
#ifdef PLSSVM_HAS_OPENCL_BACKEND
    opencl::DevicePtrOpenCL<double> r_cl(first_device, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
    //TODO: init on device
    init_(1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD), r + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMallocHost((void **)&r, dept * sizeof(double));
    cudaMalloc((void **)&r_d, (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
    init<<<1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD)>>>(r_d + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#else
    //TODO: move down

    init_(1, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD), r + dept, 0, (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    opencl::DevicePtrOpenCL<double> d_cl(first_device, dept);
    std::vector<double> toDevice(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1, 0.0);
    std::copy(b.begin(), b.begin() + dept, toDevice.begin());
    r_cl.to_device(std::vector<double>(toDevice));
    d = new double[dept];
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMallocHost((void **)&d, dept * sizeof(double));
    cudaMemcpy(r_d, &b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
#else
    //TODO:
#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    opencl::DevicePtrOpenCL<double> q_cl(first_device, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
    //TODO: init on gpu
    q_cl.to_device(std::vector<double>(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1, 0.0));
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMalloc((void **)&q_d, (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
    cudaMemset(q_d, 0, (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
#else
//TODO:
#endif
    cudaDeviceSynchronize();

#ifdef PLSSVM_HAS_OPENCL_BACKEND

    {
        if (!kernel_q_cl) {
            std::string kernel_src_file_name{"../src/OpenCL/kernels/kernel_q.cl"};
            std::string kernel_src = manager.read_src_file(kernel_src_file_name);
            json::node &deviceNode =
                manager.get_configuration()["PLATFORMS"][first_device.platformName]
                                           ["DEVICES"][first_device.deviceName];
            json::node &kernelConfig = deviceNode["KERNELS"]["kernel_q"];
            kernel_q_cl = manager.build_kernel(kernel_src, first_device, kernelConfig, "kernel_q");
        }

        const int Ncols = num_features;
        const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);

        opencl::apply_arguments(kernel_q_cl, q_cl.get(), data_cl.get(), datlast_cl.get(), Ncols, Nrows);

        size_t grid_size = ((int)dept / CUDABLOCK_SIZE + 1) * std::min((int)CUDABLOCK_SIZE, dept);
        size_t block_size = std::min((int)CUDABLOCK_SIZE, dept);
        opencl::run_kernel_1d_timed(first_device, kernel_q_cl, grid_size, block_size);
    }

#elif PLSSVM_HAS_CUDA_BACKEND
    kernel_q<<<((int)dept / CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept)>>>(q_d, data_d, datlast, num_features, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD));
#else
//TODO:
// kernel_q_(((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept),q_d, data_d, datlast, num_features , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    switch (kernel) {
    case 0: {
        if (!svm_kernel_linear) {
            std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
            std::string kernel_src = manager.read_src_file(kernel_src_file_name);
            json::node &deviceNode =
                manager.get_configuration()["PLATFORMS"][first_device.platformName]
                                           ["DEVICES"][first_device.deviceName];
            json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
            svm_kernel_linear = manager.build_kernel(kernel_src, first_device, kernelConfig, "kernel_linear");
        }

        const int Ncols = num_features;
        const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
        opencl::apply_arguments(svm_kernel_linear, q_cl.get(), r_cl.get(), x_cl.get(), data_cl.get(), QA_cost, 1 / cost, Ncols, Nrows, -1);

        std::vector<size_t> grid_size{((int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE, ((int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE};
        std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};

        opencl::run_kernel_2d_timed(first_device, svm_kernel_linear, grid_size, block_size);
    } break;
    case 1:
        //TODO: kernel_poly OpenCL
        std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
        break;
    case 2:
        //TODO: kernel_radial OpenCL
        std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
        break;
    default:
        throw std::runtime_error("Can not decide wich kernel!");
    }

#elif PLSSVM_HAS_CUDA_BACKEND
    switch (kernel) {
    case 0:
        kernel_linear<<<grid, block>>>(q_d, r_d, x_d, data_d, QA_cost, 1 / cost, num_features, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD), -1);
        break;
    case 1:
        // kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, num_features , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
        break;
    case 2:
        // kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, num_features , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
        break;
    default:
        throw std::runtime_error("Can not decide wich kernel!");
    }
#else
//TODO:
#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    {
        std::vector<double> ret(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
        r_cl.from_device(ret);
        std::copy(ret.begin(), ret.begin() + dept, r);
    }
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaDeviceSynchronize();
    cudaMemcpy(r, r_d, dept * sizeof(double), cudaMemcpyDeviceToHost);

#else
    //TODO:
#endif

    double delta = mult(r, r, dept);
    const double delta0 = delta;
    double alpha_cd, beta;
    double *Ad;

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    opencl::DevicePtrOpenCL<double> Ad_cl(first_device, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
    Ad = new double[dept];
#elif PLSSVM_HAS_CUDA_BACKEND
    double *Ad_d;
    cudaMallocHost((void **)&Ad, dept * sizeof(double));
    cudaMalloc((void **)&Ad_d, (dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
#else
    //TODO:
#endif

    int run;
    for (run = 0; run < imax; ++run) {
        if (info)
            std::cout << "Start Iteration: " << run << std::endl;
//Ad = A * d
#ifdef PLSSVM_HAS_OPENCL_BACKEND
        {
            std::vector<double> zeros(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
            Ad_cl.to_device(zeros);
            //TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
            std::vector<double> buffer(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
            r_cl.from_device(buffer);
            for (int index = dept; index < dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1; ++index)
                buffer[index] = 0.0;
            r_cl.to_device(buffer);
        }
#elif PLSSVM_HAS_CUDA_BACKEND
        cudaDeviceSynchronize();
        cudaMemset(Ad_d, 0, dept * sizeof(double));
        cudaMemset(r_d + dept, 0, ((CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
        cudaDeviceSynchronize();

#else

#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
        switch (kernel) {
        case 0: {
            if (!svm_kernel_linear) {
                std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
                std::string kernel_src = manager.read_src_file(kernel_src_file_name);
                json::node &deviceNode =
                    manager.get_configuration()["PLATFORMS"][first_device.platformName]
                                               ["DEVICES"][first_device.deviceName];
                json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
                svm_kernel_linear = manager.build_kernel(kernel_src, first_device, kernelConfig, "kernel_linear");
            }

            const int Ncols = num_features;
            const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
            opencl::apply_arguments(svm_kernel_linear, q_cl.get(), Ad_cl.get(), r_cl.get(), data_cl.get(), QA_cost, 1 / cost, Ncols, Nrows, 1);

            std::vector<size_t> grid_size{((int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE, ((int)dept / (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE};
            std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};

            opencl::run_kernel_2d_timed(first_device, svm_kernel_linear, grid_size, block_size);
        } break;
        case 1:
            //TODO: kernel_poly OpenCL
            std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
            break;
        case 2:
            //TODO: kernel_radial OpenCL
            std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
            break;
        default:
            throw std::runtime_error("Can not decide wich kernel!");
        }

#elif PLSSVM_HAS_CUDA_BACKEND
        switch (kernel) {
        case 0:
            kernel_linear<<<grid, block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1 / cost, num_features, dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD), 1);
            break;
        // case 1:
        // 	kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
        // 	break;
        // case 2:
        // 	kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
        // 	break;
        default:
            throw std::runtime_error("Can not decide wich kernel!");
        }
#else
//TODO:
#endif

#ifdef PLSSVM_HAS_OPENCL_BACKEND
        {
            std::vector<double> buffer(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
            r_cl.from_device(buffer);
            std::copy(buffer.begin(), buffer.begin() + dept, d);

            Ad_cl.from_device(buffer);
            std::copy(buffer.begin(), buffer.begin() + dept, Ad);

            std::cout << "Ad: ";
            for (double &val : std::vector<double>(Ad, Ad + dept))
                std::cout << val << " ";
            std::cout << std::endl;
        }
#elif PLSSVM_HAS_CUDA_BACKEND
        cudaDeviceSynchronize();
        cudaMemcpy(d, r_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(Ad, Ad_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Ad: ";
        for (double &val : std::vector<double>(Ad, Ad + dept))
            std::cout << val << " ";
        std::cout << std::endl;
#else

#endif

        alpha_cd = delta / mult(d, Ad, dept);

#ifdef PLSSVM_HAS_OPENCL_BACKEND
        {
            //TODO: auf GPU
            std::vector<double> buffer_r(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
            std::vector<double> buffer_x(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
            x_cl.from_device(buffer_x);
            r_cl.from_device(buffer_r);
            add_mult_(((int)dept / 1024) + 1, std::min(1024, dept), buffer_x.data(), buffer_r.data(), alpha_cd, dept);
            x_cl.to_device(buffer_x);
            std::cout << "x: ";
            for (double &val : buffer_x)
                std::cout << val << " ";
            std::cout << std::endl;
        }

#elif PLSSVM_HAS_CUDA_BACKEND
        add_mult<<<((int)dept / 1024) + 1, std::min(1024, dept)>>>(x_d, r_d, alpha_cd, dept);
        cudaDeviceSynchronize();
        std::cout << "x: ";
        for (double &val : std::vector<double>(x_d, x_d + dept))
            std::cout << val << " ";
        std::cout << std::endl;
#else
        add_mult_(((int)dept / 1024) + 1, std::min(1024, dept), x_d, r_d, alpha_cd, dept);
#endif

        // if(run%50 == 0){
        // 	cudaMemcpy(r_d, &b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
        // 	cudaDeviceSynchronize();
        // 	switch(kernel){
        // 		case 0:
        // 		kernel_linear<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
        // 		//TODO:

        // 		break;
        // 		// case 1:
        // 		// 	kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
        // 		// 	break;
        // 		// case 2:
        // 		// 	kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, num_features, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
        // 		// 	break;
        // 		default: throw std::runtime_error("Can not decide wich kernel!");
        // 	}
        // 	cudaDeviceSynchronize();
        // 	cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
        // }else{
        for (int index = 0; index < dept; ++index) {
            r[index] -= alpha_cd * Ad[index];
        }
        // }

        delta = mult(r, r, dept);
        if (delta < eps * eps * delta0)
            break;
        beta = -mult(r, Ad, dept) / mult(d, Ad, dept);
        std::cout << "delta: " << r << std::endl;
        add(mult(beta, d, dept), r, d, dept);

#ifdef PLSSVM_HAS_OPENCL_BACKEND
        {
            std::vector<double> buffer(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1, 0.0);
            std::copy(d, d + dept, buffer.begin());
            r_cl.to_device(buffer);
        }
#elif PLSSVM_HAS_CUDA_BACKEND
        cudaMemcpy(r_d, d, dept * sizeof(double), cudaMemcpyHostToDevice);
#else

#endif
    }
    if (run == imax)
        std::clog << "Regard reached maximum number of CG-iterations" << std::endl;

    alpha.resize(dept);
    std::vector<double> ret_q(dept);
    cudaDeviceSynchronize();

#ifdef PLSSVM_HAS_OPENCL_BACKEND
    {
        std::vector<double> buffer(dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) - 1);
        x_cl.from_device(buffer);
        std::copy(buffer.begin(), buffer.begin() + dept, alpha.begin());
        std::cout << "alpha: ";
        for (double &val : alpha)
            std::cout << val << " ";
        std::cout << std::endl;

        q_cl.from_device(buffer);
        std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
    }
    delete[] d, Ad;
#elif PLSSVM_HAS_CUDA_BACKEND
    cudaMemcpy(&alpha[0], x_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "alpha: ";

    for (double &val : alpha)
        std::cout << val << " ";
    std::cout << std::endl;
    cudaMemcpy(&ret_q[0], q_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(Ad_d);
    cudaFree(r_d);
    cudaFree(datlast);
    cudaFreeHost(Ad);
    cudaFree(x_d);
    cudaFreeHost(r);
    cudaFreeHost(d);
#else

#endif

    return ret_q;
}