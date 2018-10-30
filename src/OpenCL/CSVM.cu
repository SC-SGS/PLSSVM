#include "CSVM.hpp"
#include "cuda-kernel.cuh"
#include "cuda-kernel.hpp"
#include "svm-kernel.cuh"
#ifdef WITH_OPENCL
#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>


#include "../src/OpenCL/manager/apply_arguments.hpp"
#include "../src/OpenCL/manager/run_kernel.hpp"

#endif

using namespace opencl;
int CUDADEVICE = 0;

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_), kernel_q_cl(nullptr), svm_kernel_linear(nullptr){

	#ifdef WITH_OPENCL
	
		
		std::vector<opencl::device_t> &devices = manager.get_devices();
		
		/*if (devices.size() > 1) {
			throw std::runtime_error("can only use a single device for OpenCL version");
		}*/
		first_device = devices[0];
	
		
		
	#endif
	}

void CSVM::learn(){
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
	
	if(info)std::cout << "start CG" << std::endl;
	//solve minimization
	q = CG(b,Nfeatures_data,epsilon);
    alpha.emplace_back(-sum(alpha));
	bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}


double CSVM::kernel_function(std::vector<double>& xi, std::vector<double>& xj){
	switch(kernel){
		case 0: return xi * xj;
		case 1: return std::pow(gamma * (xi*xj) + coef0 ,degree);
		case 2: {double temp = 0;
			for(int i = 0; i < xi.size(); ++i){
				temp += (xi-xj)*(xi-xj);
			}
			return exp(-gamma * temp);}
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	
}


void CSVM::loadDataDevice(){
	cudaMallocManaged((void **) &datlast, (Nfeatures_data + CUDABLOCK_SIZE - 1) * sizeof(double));
	cudaMemset(datlast, 0, Nfeatures_data + CUDABLOCK_SIZE - 1 * sizeof(double));
	cudaMemcpy(datlast,&data[Ndatas_data - 1][0], Nfeatures_data * sizeof(double), cudaMemcpyHostToDevice);
	cudaMallocManaged((void **) &data_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double));

	double* col_vec;
	cudaMallocHost((void **) &col_vec, (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)  * sizeof(double));
	
	#pragma parallel for
	for(size_t col = 0; col < Nfeatures_data ; ++col){
		for(size_t row = 0; row < Ndatas_data - 1; ++row){
			col_vec[row] = data[row][col];
		}
		for(int i = 0 ; i < + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) ; ++i){
			col_vec[i + Ndatas_data - 1] = 0;
		}
		cudaMemcpy(data_d + col * (Ndatas_data+ (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1), col_vec, (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1) *sizeof(double), cudaMemcpyHostToDevice);
	}
	cudaFreeHost(col_vec);
}
/*
void CSVM::loadDataDevice(){
	//cudaMalloc((void **) &datlast, (Nfeatures_data + CUDABLOCK_SIZE - 1) * sizeof(double));
	datlast = new double[Nfeatures_data + CUDABLOCK_SIZE - 1];
	//cudaMemset(datlast, 0, Nfeatures_data + CUDABLOCK_SIZE - 1 * sizeof(double));
	for(int i = 0; i < Nfeatures_data + CUDABLOCK_SIZE - 1; ++i) datlast[i] = 0;
	
	//cudaMemcpy(datlast,&data[Ndatas_data - 1][0], Nfeatures_data * sizeof(double), cudaMemcpyHostToDevice);
	for(int i = 0; i < Nfeatures_data; ++i) datlast[i] = data[Ndatas_data - 1][i];
	//datlast.to_device(data[Ndatas_data - 1]);

	//cudaMalloc((void **) &data_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double));
	data_d = new double[Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)];
	//opencl::DevicePtrOpenCL<double> datlast = DevicePtrOpenCL<double>(device, (Nfeatures_data + CUDABLOCK_SIZE - 1));

	double* col_vec;
	//cudaMallocHost((void **) &col_vec, (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)  * sizeof(double));
	col_vec = new double[Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1];

	#pragma parallel for
	for(size_t col = 0; col < Nfeatures_data ; ++col){
		for(size_t row = 0; row < Ndatas_data - 1; ++row){
			col_vec[row] = data[row][col];
		}
		for(int i = 0 ; i < + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) ; ++i){
			col_vec[i + Ndatas_data - 1] = 0;
		}
		//cudaMemcpy(data_d + col * (Ndatas_data+ (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1), col_vec, (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1) *sizeof(double), cudaMemcpyHostToDevice);
		for (int i = 0 ; i < Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1; ++i) data_d[col * (Ndatas_data+ (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1) + i] = col_vec[i];
	}
	
	//cudaFreeHost(col_vec);
	delete [] col_vec;
}*/


void CSVM::learn(std::string &filename, std::string &output_filename) {
	auto begin_parse = std::chrono::high_resolution_clock::now();
	if(filename.size() > 5 && endsWith(filename, ".arff")){
		arffParser(filename);
	}else{
		libsvmParser(filename);
	}

	auto end_parse = std::chrono::high_resolution_clock::now();
	if(info){std::clog << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data  <<" in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << " ms eingelesen" << std::endl << std::endl ;}
	
	cudaSetDevice(CUDADEVICE);
	loadDataDevice();
	
	auto end_gpu = std::chrono::high_resolution_clock::now();
	
	if(info) std::clog << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count() << " ms auf die Gpu geladen" << std::endl << std::endl ;

	learn();
	auto end_learn = std::chrono::high_resolution_clock::now();
    if(info) std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << " ms gelernt" << std::endl;

	writeModel(output_filename);
	auto end_write = std::chrono::high_resolution_clock::now();
    if(info){std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << " geschrieben" << std::endl;
    }else if(times){
		std::clog << data.size()<<", "<< Nfeatures_data  <<", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", "<< std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count()<< ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << std::endl;
	} 

}


std::vector<double>CSVM::CG(const std::vector<double> &b,const int imax,  const double eps)
{
	const int dept = Ndatas_data - 1;

	// #ifdef WITH_OPENCL
	//TODO
	// #elif WITH_CUDA
	dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	dim3 block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
	// #else
	//TODO
	// #endif


	double *x_d, *r, *d, *r_d, *q_d, *x;
	
	// #ifdef WITH_OPENCL
	opencl::DevicePtrOpenCL<double> x_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #elif WITH_CUDA
	cudaMallocManaged((void **) &x_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)*sizeof(double));
	// #else
	x = new double[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];
	

	// #endif
	

	// #ifdef WITH_OPENCL
	//TODO
	init_(((int) dept/1024) + 1, std::min(1024, dept),x,1,dept);
	init_(1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1,x + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	x_cl.to_device(std::vector<double>(x, x+(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)));
	// #elif WITH_CUDA
	init<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,1,dept);
	init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1>>>(x_d + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #else
	init_(((int) dept/1024) + 1, std::min(1024, dept),x,1,dept);
	init_(1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1,x + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #endif

	cudaDeviceSynchronize();
	
	#ifdef WITH_OPENCL
	//TODO: richtiges Device einlesen
	opencl::DevicePtrOpenCL<double> data_cl(first_device, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1));
	data_cl.to_device(std::vector<double>(data_d, data_d +  Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)));
	opencl::DevicePtrOpenCL<double> datlast_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	datlast_cl.to_device(std::vector<double>(datlast, datlast + dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1));
	#elif WITH_CUDA
	//TODO:
	#else
	//TODO:
	#endif

	r = new double[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];

	//r = b - (A * x)
	///r = b;
	// #ifdef WITH_OPENCL
	opencl::DevicePtrOpenCL<double> r_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on device
	init_( 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), r + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #elif WITH_CUDA
	cudaMallocHost((void **) &r, dept *sizeof(double));
	cudaMallocManaged((void **) &r_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) *sizeof(double));
	init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD)>>>(r_d + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #else
	//TODO: move down
	
	init_( 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), r + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	// #endif




	// #ifdef WITH_OPENCL
	opencl::DevicePtrOpenCL<double> d_cl(first_device, dept);
	std::vector<double> toDevice(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
	std::copy(b.begin(), b.begin() + dept, toDevice.begin());
	r_cl.to_device(std::vector<double>(toDevice));
	// #elif WITH_CUDA
	cudaMallocHost((void **) &d, dept *sizeof(double));
	cudaMemcpy(r_d,&b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
	// #else
	//TODO:
	// #endif
	


	// #ifdef WITH_OPENCL
	opencl::DevicePtrOpenCL<double> q_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on gpu
	q_cl.to_device(std::vector<double>(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0));
	// #elif WITH_CUDA
	cudaMallocManaged((void **) &q_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
	cudaMemset(q_d, 0, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  * sizeof(double));
	// #else
	//TODO:
	// #endif
	cudaDeviceSynchronize();

	#ifdef WITH_OPENCL

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


		const int Ncols = Nfeatures_data;
		const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
		
	  	opencl::apply_arguments(kernel_q_cl, q_cl.get(), data_cl.get(), datlast_cl.get(), Ncols , Nrows);

    size_t grid_size = ((int) dept/CUDABLOCK_SIZE + 1) *  std::min((int)CUDABLOCK_SIZE, dept) ;
    size_t block_size = std::min((int)CUDABLOCK_SIZE, dept);
    opencl::run_kernel_1d_timed(first_device, kernel_q_cl, grid_size, block_size);
	}

	//#elif WITH_CUDA
	kernel_q<<<((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept)>>>(q_d, data_d, datlast, Nfeatures_data , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
	// #else
	//TODO:
	// kernel_q_(((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept),q_d, data_d, datlast, Nfeatures_data , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
	#endif



	#ifdef WITH_OPENCL
	switch(kernel){
		case 0: 
			{
				if (!svm_kernel_linear) {
					std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
					std::string kernel_src = manager.read_src_file(kernel_src_file_name);
					json::node &deviceNode =
						manager.get_configuration()["PLATFORMS"][first_device.platformName]
												["DEVICES"][first_device.deviceName];
					json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
					svm_kernel_linear = manager.build_kernel(kernel_src, first_device, kernelConfig, "kernel_linear");
				}
		
				
				const int Ncols = Nfeatures_data;
				const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
				opencl::apply_arguments(svm_kernel_linear, q_cl.get(), r_cl.get(), x_cl.get(), data_cl.get(), QA_cost , 1/cost, Ncols, Nrows, -1);
	
			std::vector<size_t> grid_size{((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE ,((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE};
			std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};
			
			opencl::run_kernel_2d_timed(first_device, svm_kernel_linear, grid_size, block_size);
			}
			break;
		case 1: 
			//TODO: kernel_poly OpenCL
			std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
			break;	
	   case 2: 
		   //TODO: kernel_radial OpenCL
			std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
			break;	
	   default: throw std::runtime_error("Can not decide wich kernel!");
   }

   
	//  #elif WITH_CUDA
	switch(kernel){
		case 0: 
			kernel_linear<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
			break;
		case 1: 
		 	// kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
		 	break;	
		case 2: 
			// kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
			break;
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	#else
	//TODO:
	#endif

	


	#ifdef WITH_OPENCL
		{
			std::vector<double> ret(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
			r_cl.from_device(ret);
			std::copy(ret.begin(), ret.begin() + dept, r);
		}
	#elif WITH_CUDA
		cudaDeviceSynchronize();
		cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);

	#else
		//TODO:
	#endif
	
	
	
	double delta = mult(r, r, dept);	
	const double delta0 = delta;
	double alpha_cd, beta;
	double* Ad;

	#ifdef WITH_OPENCL
		opencl::DevicePtrOpenCL<double> Ad_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);

	// #elif WITH_CUDA
		double *Ad_d; 
		cudaMallocHost((void **) &Ad, dept *sizeof(double));
		cudaMallocManaged((void **) &Ad_d, (dept +(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  *sizeof(double));
	#else
		//TODO:
	#endif
	

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d


		



		cudaDeviceSynchronize();




		#ifdef WITH_OPENCL
			{
				std::vector<double> zeros( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
				Ad_cl.to_device(zeros);
				//TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
				std::vector<double> buffer( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
				r_cl.from_device(buffer);
				for(int index = dept; index <  dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++index) buffer[index] = 0.0;
				r_cl.to_device(buffer);
			}
		// #elif WITH_CUDA
			cudaMemset(Ad_d, 0, dept * sizeof(double));
			cudaMemset(r_d + dept, 0, ((CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
			cudaDeviceSynchronize();

		#else

		#endif



		#ifdef WITH_OPENCL
		switch(kernel){
			case 0: 
				{
					if (!svm_kernel_linear) {
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						json::node &deviceNode =
							manager.get_configuration()["PLATFORMS"][first_device.platformName]
													["DEVICES"][first_device.deviceName];
						json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
						svm_kernel_linear = manager.build_kernel(kernel_src, first_device, kernelConfig, "kernel_linear");
					}
			
					
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
					opencl::apply_arguments(svm_kernel_linear, q_cl.get(), Ad_cl.get(), r_cl.get(), data_cl.get(), QA_cost , 1/cost, Ncols, Nrows, 1);
		
				std::vector<size_t> grid_size{((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE ,((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE};
				std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};
				
				opencl::run_kernel_2d_timed(first_device, svm_kernel_linear, grid_size, block_size);
				}
				break;
			case 1: 
				//TODO: kernel_poly OpenCL
				std::cerr << "kernel_poly not jet implemented in OpenCl use CUDA";
				break;	
		case 2: 
			//TODO: kernel_radial OpenCL
				std::cerr << "kernel_radial not jet implemented in OpenCl use CUDA";
				break;	
		default: throw std::runtime_error("Can not decide wich kernel!");
	}

	
		//  #elif WITH_CUDA
		switch(kernel){
			case 0: 
				kernel_linear<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1);
				break;
			// case 1: 
			// 	kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
			// 	break;
			// case 2: 
			// 	kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
			// 	break;
			default: throw std::runtime_error("Can not decide wich kernel!");
		}
		#else
		//TODO:
		#endif
		





	







		#ifdef WITH_OPENCL
		{
			std::vector<double> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
			r_cl.from_device(buffer);
			std::copy(buffer.begin(), buffer.begin()+dept, d);

			Ad_cl.from_device(buffer);
			std::copy(buffer.begin(), buffer.begin()+dept, Ad);

			std::cout << "Ad: ";
			for(double & val : std::vector<double>(d, d+dept)) std::cout << val << " "; std::cout << std::endl;

		}
		//  #elif WITH_CUDA
		cudaDeviceSynchronize();
		cudaMemcpy(d, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ad, Ad_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		std::cout << "Ad: ";
			for(double & val : std::vector<double>(d, d+dept)) std::cout << val << " "; std::cout << std::endl;
		#else

		#endif


		

		
		alpha_cd = delta / mult(d , Ad,  dept);
		//add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
		add_mult_(((int) dept/1024) + 1, std::min(1024, dept),x_d,r_d,alpha_cd,dept);
		if(run%50 == 0){
			cudaMemcpy(r_d, &b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			switch(kernel){
				case 0: 
					kernel_linear<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
					break;
				// case 1: 
				// 	kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
				// 	break;
				// case 2: 
				// 	kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
				// 	break;
				default: throw std::runtime_error("Can not decide wich kernel!");
			}
			cudaDeviceSynchronize();	
			cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		}else{
			for(int index = 0; index < dept; ++index){
				r[index] -= alpha_cd * Ad[index];
			}
		}
		delta = mult(r , r, dept);
		if(delta < eps * eps * delta0) break;
		beta = -mult(r, Ad, dept) / mult(d, Ad, dept);
		std::cout << "delta: "<< r << std::endl;
		add(mult(beta, d, dept),r, d, dept);

		#ifdef WITH_OPENCL
		{
			std::vector<double> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
			std::copy(d, d+dept, buffer.begin());
			r_cl.to_device(buffer);

		}
		// #elif WITH_CUDA
			cudaMemcpy(r_d, d, dept*sizeof(double), cudaMemcpyHostToDevice);
		#else

		#endif

	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
	alpha.resize(dept);
	std::vector<double> ret_q(dept);
	cudaDeviceSynchronize();
	cudaMemcpy(&alpha[0],x_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&ret_q[0],q_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(Ad_d);
	cudaFree(r_d);
	cudaFree(datlast);
	cudaFreeHost(Ad);
	cudaFree(x_d);
	cudaFreeHost(r);
	cudaFreeHost(d);
	return ret_q;
}