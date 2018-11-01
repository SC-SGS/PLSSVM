#include "CSVM.hpp"
#include "cuda-kernel.hpp"

#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>


#include "../src/OpenCL/manager/apply_arguments.hpp"
#include "../src/OpenCL/manager/run_kernel.hpp"



int CUDADEVICE = 0;

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_), kernel_q_cl(nullptr), svm_kernel_linear(nullptr){
	std::vector<opencl::device_t> &devices = manager.get_devices();
	first_device = devices[0];

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
	datlast_cl = opencl::DevicePtrOpenCL<double> (first_device, (Nfeatures_data + CUDABLOCK_SIZE - 1));
	std::vector<double> datalast(data[Ndatas_data - 1]);
	for(int i = 0; i < CUDABLOCK_SIZE - 1 ; ++i )datalast.push_back( 0.0);
	datlast_cl.to_device(datalast);


	data_cl  = opencl::DevicePtrOpenCL<double>(first_device, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1));
	std::vector<double> vec;
	vec.reserve(Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1);
	// #pragma parallel for 
	for(size_t col = 0; col < Nfeatures_data ; ++col){
		for(size_t row = 0; row < Ndatas_data - 1; ++row){
			vec.push_back(data[row][col]);
		}
		for(int i = 0 ; i < + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) ; ++i){
			vec.push_back(0.0);
		}
	}
	data_cl.to_device(vec);
}


void CSVM::learn(std::string &filename, std::string &output_filename) {
	auto begin_parse = std::chrono::high_resolution_clock::now();
	if(filename.size() > 5 && endsWith(filename, ".arff")){
		arffParser(filename);
	}else{
		libsvmParser(filename);
	}

	auto end_parse = std::chrono::high_resolution_clock::now();
	if(info){std::clog << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data  <<" in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << " ms eingelesen" << std::endl << std::endl ;}
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

	#ifdef WITH_OPENCL
	//TODO: dim und grid
	#elif WITH_CUDA
	dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	dim3 block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
	double *x_d, *r_d, *q_d;
	#else
	//TODO
	#endif


	double *x, *r, *d;
	opencl::DevicePtrOpenCL<double> x_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	x = new double[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];

	
	//TODO: init on GPU
	init_(((int) dept/1024) + 1, std::min(1024, dept),x,1,dept);
	init_(1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1,x + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	x_cl.to_device(std::vector<double>(x, x+(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)));
	

	r = new double[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];

	//r = b - (A * x)
	///r = b;
	opencl::DevicePtrOpenCL<double> r_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on device
	init_( 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), r + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	

	opencl::DevicePtrOpenCL<double> d_cl(first_device, dept);
	std::vector<double> toDevice(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
	std::copy(b.begin(), b.begin() + dept, toDevice.begin());
	r_cl.to_device(std::vector<double>(toDevice));
	d = new double[dept];

	opencl::DevicePtrOpenCL<double> q_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on gpu
	q_cl.to_device(std::vector<double>(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0));

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


	{
		std::vector<double> ret(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		r_cl.from_device(ret);
		std::copy(ret.begin(), ret.begin() + dept, r);
	}
	
	
	
	double delta = mult(r, r, dept);	
	const double delta0 = delta;
	double alpha_cd, beta;
	double* Ad;


	opencl::DevicePtrOpenCL<double> Ad_cl(first_device, dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	Ad = new double[dept];

	

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		{
			std::vector<double> zeros( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
			Ad_cl.to_device(zeros);
			//TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
			std::vector<double> buffer( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
			r_cl.from_device(buffer);
			for(int index = dept; index <  dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++index) buffer[index] = 0.0;
			r_cl.to_device(buffer);
		}

		


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

	

		{
			std::vector<double> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
			r_cl.from_device(buffer);
			std::copy(buffer.begin(), buffer.begin()+dept, d);

			Ad_cl.from_device(buffer);
			std::copy(buffer.begin(), buffer.begin()+dept, Ad);

		}
	


		
		
		
		alpha_cd = delta / mult(d , Ad,  dept);
	

		
		
		//TODO: auf GPU
		std::vector<double> buffer_r(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		std::vector<double> buffer_x(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		x_cl.from_device(buffer_x);
		r_cl.from_device(buffer_r);
		add_mult_(((int) dept/1024) + 1, std::min(1024, dept),buffer_x.data(),buffer_r.data(),alpha_cd,dept);
		x_cl.to_device(buffer_x);
		

		


		if(run%50 == 0){
			std::vector<double> buffer(b);
			for(int i = 0; i <  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++ i) buffer.push_back(0.0);
			r_cl.to_device(buffer);

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
	
			r_cl.from_device(buffer);
			std::copy(buffer.begin(), buffer.begin()+dept, r);
		}else{
			for(int index = 0; index < dept; ++index){
				r[index] -= alpha_cd * Ad[index];
			}
		}
		
		delta = mult(r , r, dept);
		if(delta < eps * eps * delta0) break;
		beta = -mult(r, Ad, dept) / mult(d, Ad, dept);
		add(mult(beta, d, dept),r, d, dept);


		{
			std::vector<double> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
			std::copy(d, d+dept, buffer.begin());
			r_cl.to_device(buffer);

		}
		
		
	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
	
	
	alpha.resize(dept);
	std::vector<double> ret_q(dept);


	{
		std::vector<double> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		x_cl.from_device(buffer);
		std::copy(buffer.begin(), buffer.begin() + dept, alpha.begin());
		q_cl.from_device(buffer);
		std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
	}
	delete[] d, Ad;
	return ret_q;
}