#include "CSVM.hpp"
#include "cuda-kernel.hpp"

#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>


#include "../src/OpenCL/manager/apply_arguments.hpp"
#include "../src/OpenCL/manager/run_kernel.hpp"




int count_devices = 1;


CSVM::CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){
	std::vector<opencl::device_t> &devices = manager.get_devices();
	first_device = devices[0];
	count_devices = devices.size();
	svm_kernel_linear.resize(count_devices, nullptr);
	kernel_q_cl.resize(count_devices, nullptr);
	std::cout << "GPUs found: " << count_devices << std::endl;
	}

void CSVM::learn(){
	std::vector<real_t> q;
	std::vector<real_t> b = value;

	b.pop_back();
	b -= value.back();

	QA_cost = kernel_function(data.back(), data.back()) + 1 / cost;

	
	if(info)std::cout << "start CG" << std::endl;
	//solve minimization
	q = CG(b,Nfeatures_data,epsilon);
    alpha.emplace_back(-sum(alpha));
	bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}


real_t CSVM::kernel_function(std::vector<real_t>& xi, std::vector<real_t>& xj){
	switch(kernel){
		case 0: return xi * xj;
		case 1: return std::pow(gamma * (xi*xj) + coef0 ,degree);
		case 2: {real_t temp = 0;
			for(int i = 0; i < xi.size(); ++i){
				temp += (xi-xj)*(xi-xj);
			}
			return exp(-gamma * temp);}
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	
}


void CSVM::loadDataDevice(){
	std::vector<opencl::device_t> &devices = manager.get_devices(); //TODO: header
	for(int i = 0; i < count_devices; ++i) datlast_cl.emplace_back(opencl::DevicePtrOpenCL<real_t> (devices[i], (Nfeatures_data)));
	std::vector<real_t> datalast(data[Ndatas_data - 1]);

	for(int i = 0; i < count_devices; ++i) datlast_cl[i].to_device(datalast);
	for(int i = 0; i < count_devices; ++i) datlast_cl[i].resize(Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);

	for(int i = 0; i < count_devices; ++i) data_cl.emplace_back(opencl::DevicePtrOpenCL<real_t>(devices[i], Nfeatures_data * (Ndatas_data - 1)));
	for(int i = 0; i < count_devices; ++i) resizeData(i,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);	
		
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





void CSVM::resizeData(const int device, const int boundary){
	std::vector<opencl::device_t> &devices = manager.get_devices(); //TODO: header

	data_cl[device] = opencl::DevicePtrOpenCL<real_t>(devices[device], Nfeatures_data * (Ndatas_data - 1 + boundary));
	std::vector<real_t> vec;
	//vec.reserve(Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1);
	for(size_t col = 0; col < Nfeatures_data ; ++col){
		for(size_t row = 0; row < Ndatas_data - 1; ++row){
			vec.push_back(data[row][col]);
		}
		for(int i = 0 ; i < boundary ; ++i){
			vec.push_back(0.0);
		}
	}
	data_cl[device].to_device(vec);
}



std::vector<real_t>CSVM::CG(const std::vector<real_t> &b,const int imax,  const real_t eps)
{
	std::vector<opencl::device_t> &devices = manager.get_devices(); //TODO: header
	const size_t dept = Ndatas_data - 1;
	const size_t boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
	const size_t dept_all = dept + boundary_size;
	std::vector<real_t> zeros(dept_all, 0.0);

	real_t *d;
	std::vector<real_t> x(dept_all , 1.0);
	std::fill(x.end() - boundary_size, x.end(), 0.0);

	std::vector<opencl::DevicePtrOpenCL<real_t> > x_cl;
	x_cl.emplace_back( devices[0] , dept_all);
	x_cl[0].to_device(x);

	
	
	std::vector<real_t> r(dept_all, 0.0);

	std::vector<opencl::DevicePtrOpenCL<real_t> > r_cl;
	r_cl.emplace_back(devices[0], dept_all);
	
;
	std::vector<real_t> toDevice(dept_all, 0.0);
	std::copy(b.begin(), b.begin() + dept, toDevice.begin());
	r_cl[0].to_device(std::vector<real_t>(toDevice));
	// r_cl[0].to_device(std::vector<real_t>(zeros));
	d = new real_t[dept];

	std::vector<opencl::DevicePtrOpenCL<real_t> > q_cl;
	q_cl.emplace_back( devices[0], dept_all);
	//TODO: init on gpu
	q_cl[0].to_device(std::vector<real_t>(dept_all, 0.0));


		if (!kernel_q_cl[0]) {
			std::string kernel_src_file_name{"../src/OpenCL/kernels/kernel_q.cl"};
			std::string kernel_src = manager.read_src_file(kernel_src_file_name);
			json::node &deviceNode =
				manager.get_configuration()["PLATFORMS"][devices[0].platformName]
										["DEVICES"][devices[0].deviceName];
			json::node &kernelConfig = deviceNode["KERNELS"]["kernel_q"];
			kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
			kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
			kernel_q_cl[0] = manager.build_kernel(kernel_src, devices[0], kernelConfig, "kernel_q");
		}
		const int Ncols = Nfeatures_data;
		const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
		
	  	opencl::apply_arguments(kernel_q_cl[0], q_cl[0].get(), data_cl[0].get(), datlast_cl[0].get(), Ncols , Nrows);

		

		size_t grid_size = ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE)) * THREADBLOCK_SIZE;
		size_t block_size = THREADBLOCK_SIZE ;
		opencl::run_kernel_1d_timed(devices[0], kernel_q_cl[0], grid_size, block_size);
	
		
	switch(kernel){
		case 0: 
			//#pragma omp parallel for
			  
				if (!svm_kernel_linear[0]) {
					std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
					std::string kernel_src = manager.read_src_file(kernel_src_file_name);
					json::node &deviceNode =
						manager.get_configuration()["PLATFORMS"][devices[0].platformName]
												["DEVICES"][devices[0].deviceName];
					json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
					kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
					kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
					svm_kernel_linear[0] = manager.build_kernel(kernel_src, devices[0], kernelConfig, "kernel_linear");

				}
				{
					 q_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					r_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					x_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					resizeData(0,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
					opencl::apply_arguments(svm_kernel_linear[0], q_cl[0].get(), r_cl[0].get(), x_cl[0].get(), data_cl[0].get(), QA_cost , 1/cost, Ncols, Nrows, -1, 0, 0);
					std::vector<size_t> grid_size{ ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))* THREADBLOCK_SIZE,ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)) * THREADBLOCK_SIZE};
					std::cout << grid_size[0] << "; " << grid_size[1]<< std::endl;
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};

					opencl::run_kernel_2d_timed(devices[0], svm_kernel_linear[0], grid_size, block_size);
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
// 	{
// 	   std::vector<double> buffer(dept_all );
// 	   r_cl[0].from_device(buffer);
// 	   for(auto value: buffer){
// 		   std::cout << value << " ";
// 	   }
// 	   std::cout << std::endl;
//    }
// exit(0);
		r_cl[0].resize(dept_all);
		r_cl[0].from_device(r);

	real_t delta = mult(r.data(), r.data(), dept); //TODO:	
	const real_t delta0 = delta;
	real_t alpha_cd, beta;
	real_t* Ad;


	std::vector<opencl::DevicePtrOpenCL<real_t> > Ad_cl;
	Ad_cl.emplace_back(devices[0], dept_all);
	Ad = new real_t[dept];

	

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d

		Ad_cl[0].to_device(zeros);
		//TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )

		std::vector<real_t> buffer( dept_all);
		r_cl[0].resize(dept_all);
		r_cl[0].from_device(buffer);
		for(int index = dept; index < dept_all; ++index) buffer[index] = 0.0;
		r_cl.resize(dept_all);
		r_cl[0].to_device(buffer);
	


		switch(kernel){
			case 0: 
				// #pragma omp parallel for
					{
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						json::node &deviceNode =
							manager.get_configuration()["PLATFORMS"][devices[0].platformName]
													["DEVICES"][devices[0].deviceName];
						json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
						kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
						kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
						svm_kernel_linear[0] = manager.build_kernel(kernel_src, devices[0], kernelConfig, "kernel_linear");
			

					q_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					Ad_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);  
					r_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					resizeData(0,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
					opencl::apply_arguments(svm_kernel_linear[0], q_cl[0].get(), Ad_cl[0].get(), r_cl[0].get(), data_cl[0].get(), QA_cost , 1/cost, Ncols, Nrows, 1, 0, 0);
					std::vector<size_t> grid_size{ ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)) * THREADBLOCK_SIZE, ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))  * THREADBLOCK_SIZE};
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};

					opencl::run_kernel_2d_timed(devices[0], svm_kernel_linear[0], grid_size, block_size);
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


		
		for(int i = 0; i< dept; ++i) d[i] = r[i];

		 
		std::vector<real_t> ret(dept_all);
		Ad_cl.resize(dept_all);
		Ad_cl[0].from_device(ret);
		std::copy(ret.begin(), ret.begin()+dept, Ad);
		

		alpha_cd = delta / mult(d , Ad,  dept);
		
		//TODO: auf GPU
		std::vector<real_t> buffer_r(dept_all );
		r_cl[0].resize(dept_all);
		r_cl[0].from_device(buffer_r);
		add_mult_(((int) dept/1024) + 1, std::min(1024, (int) dept),x.data(),buffer_r.data(),alpha_cd,dept);
		x_cl.resize(dept_all);
		x_cl[0].to_device(x);

		

		

		if(run%50 == 0){
			std::vector<real_t> buffer(b);
			for(int i = 0; i <  THREADBLOCK_SIZE * INTERNALBLOCK_SIZE; ++ i) buffer.push_back(0.0); //TODO: resize
			r_cl.resize(dept_all);
			r_cl[0].to_device(buffer);

			switch(kernel){
			case 0: 
				// #pragma omp parallel for
					if (!svm_kernel_linear[0]) {
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						json::node &deviceNode =
							manager.get_configuration()["PLATFORMS"][devices[0].platformName]
													["DEVICES"][devices[0].deviceName];
						json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
						kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(INTERNALBLOCK_SIZE));
						kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(THREADBLOCK_SIZE));
						svm_kernel_linear[0] = manager.build_kernel(kernel_src, devices[0], kernelConfig, "kernel_linear");
					}
			
					

					{
					q_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					r_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					x_cl[0].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					resizeData(0,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
					opencl::apply_arguments(svm_kernel_linear[0], q_cl[0].get(), r_cl[0].get(), x_cl[0].get(), data_cl[0].get(), QA_cost , 1/cost, Ncols, Nrows, -1), 0,0;
					std::vector<size_t> grid_size{ ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)) * THREADBLOCK_SIZE, ceil(static_cast<double>(dept) / static_cast<double>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))  * THREADBLOCK_SIZE};
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};
					
					opencl::run_kernel_2d_timed(devices[0], svm_kernel_linear[0], grid_size, block_size);
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
				r_cl.resize(dept_all);
				r_cl[0].from_device(r);
				r_cl[0].to_device(r);
			}
		}else{
			for(int index = 0; index < dept; ++index){
				r[index] -= alpha_cd * Ad[index];
			}
		}
		
		delta = mult(r.data() , r.data(), dept); //TODO:
		if(delta < eps * eps * delta0) break;
		beta = -mult(r.data(), Ad, dept) / mult(d, Ad, dept); //TODO:
		add(mult(beta, d, dept),r.data(), d, dept);//TODO:


		{
			std::vector<real_t> buffer(dept_all, 0.0);
			std::copy(d, d+dept, buffer.begin());
			r_cl.resize(dept_all);
			r_cl[0].to_device(buffer);

		}
		
		
	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
	
	
	alpha.resize(dept);
	std::vector<real_t> ret_q(dept);


	{
		std::vector<real_t> buffer(dept_all );
		std::copy(x.begin(), x.begin() + dept, alpha.begin());
		q_cl[0].resize(dept_all);
		q_cl[0].from_device(buffer);
		std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
	}
	delete[] d, Ad;
	return ret_q;
}