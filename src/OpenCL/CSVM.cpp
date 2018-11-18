#include "CSVM.hpp"
#include "cuda-kernel.hpp"

#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>


#include "../src/OpenCL/manager/apply_arguments.hpp"
#include "../src/OpenCL/manager/run_kernel.hpp"



int CUDADEVICE = 1;
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
	for(int i = 0; i < count_devices; ++i) datlast_cl.emplace_back(opencl::DevicePtrOpenCL<real_t> (devices[i], (Nfeatures_data + CUDABLOCK_SIZE - 1)));
	std::vector<real_t> datalast(data[Ndatas_data - 1]);
	for(int i = 0; i < CUDABLOCK_SIZE - 1 ; ++i )datalast.push_back( 0.0);
	for(int i = 0; i < count_devices; ++i) datlast_cl[i].to_device(datalast);

	for(int i = 0; i < count_devices; ++i) data_cl.emplace_back(opencl::DevicePtrOpenCL<real_t>(devices[i], Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)));
	std::vector<real_t> vec;
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
	for(int i = 0; i < count_devices; ++i) data_cl[i].to_device(vec);
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


std::vector<real_t>CSVM::CG(const std::vector<real_t> &b,const int imax,  const real_t eps)
{
	std::vector<opencl::device_t> &devices = manager.get_devices(); //TODO: header
	const int dept = Ndatas_data - 1;

	real_t *x, *r, *d;
	std::vector<opencl::DevicePtrOpenCL<real_t> > x_cl;
	for(int i = 0; i < count_devices; ++i) x_cl.emplace_back( devices[i] , dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	x = new real_t[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];

	
	//TODO: init on GPU
	init_(((int) dept/1024) + 1, std::min(1024, dept),x,1,dept);
	init_(1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1,x + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	
	for(int i = 0; i < count_devices; ++i) x_cl[i].to_device(std::vector<real_t>(x, x+(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)));
	

	r = new real_t[(dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)];

	//r = b - (A * x)
	///r = b;
	std::vector<opencl::DevicePtrOpenCL<real_t> > r_cl;
	for(int i = 0; i < count_devices; ++i) r_cl.emplace_back(devices[i], dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on device
	init_( 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), r + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	

	std::vector<opencl::DevicePtrOpenCL<real_t> > d_cl;
	for(int i = 0; i < count_devices; ++i) d_cl.emplace_back(devices[i], dept);
	std::vector<real_t> toDevice(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
	std::copy(b.begin(), b.begin() + dept, toDevice.begin());
	for(int i = 0; i < count_devices; ++i) r_cl[i].to_device(std::vector<real_t>(toDevice));
	d = new real_t[dept];

	std::vector<opencl::DevicePtrOpenCL<real_t> > q_cl;
	for(int i = 0; i < count_devices; ++i) q_cl.emplace_back( devices[i], dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//TODO: init on gpu
	for(int i = 0; i < count_devices; ++i) q_cl[i].to_device(std::vector<real_t>(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0));

	#pragma omp parallel for
	for(int i = 0; i < count_devices; ++i){
		if (!kernel_q_cl[i]) {
			std::string kernel_src_file_name{"../src/OpenCL/kernels/kernel_q.cl"};
			std::string kernel_src = manager.read_src_file(kernel_src_file_name);
			json::node &deviceNode =
				manager.get_configuration()["PLATFORMS"][devices[i].platformName]
										["DEVICES"][devices[i].deviceName];
			json::node &kernelConfig = deviceNode["KERNELS"]["kernel_q"];
			kernel_q_cl[i] = manager.build_kernel(kernel_src, devices[i], kernelConfig, "kernel_q");
		}
		const int Ncols = Nfeatures_data;
		const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
		
	  	opencl::apply_arguments(kernel_q_cl[i], q_cl[i].get(), data_cl[i].get(), datlast_cl[i].get(), Ncols , Nrows);

		size_t grid_size = ((int) dept/CUDABLOCK_SIZE + 1) *  std::min((int)CUDABLOCK_SIZE, dept) ;
		size_t block_size = std::min((int)CUDABLOCK_SIZE, dept);
		opencl::run_kernel_1d_timed(devices[i], kernel_q_cl[i], grid_size, block_size);
	}




	switch(kernel){
		case 0: 
			//#pragma omp parallel for
			for(int i = 0; i < count_devices; ++i){
				if (!svm_kernel_linear[i]) {
					std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
					std::string kernel_src = manager.read_src_file(kernel_src_file_name);
					json::node &deviceNode =
						manager.get_configuration()["PLATFORMS"][devices[i].platformName]
												["DEVICES"][devices[i].deviceName];
					json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
					svm_kernel_linear[i] = manager.build_kernel(kernel_src, devices[i], kernelConfig, "kernel_linear");
				}
		
				const int Ncols = Nfeatures_data;
				const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
				// std::vector<size_t> grid_size{((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1)) * CUDABLOCK_SIZE ,((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1)) * CUDABLOCK_SIZE};
				std::vector<size_t> grid_size{ (int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1), (int)((dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) / count_devices)};
				//TODO: bessere aufteilung (tuning parameter?)
				// opencl::apply_arguments(svm_kernel_linear[i], q_cl[i].get(), r_cl[i].get(), x_cl[i].get(), data_cl[i].get(), QA_cost , 1/cost, Ncols, Nrows, -1, 0, 0);
				opencl::apply_arguments(svm_kernel_linear[i], q_cl[i].get(), r_cl[i].get(), x_cl[i].get(), data_cl[i].get(), QA_cost , 1/cost, Ncols, Nrows, -1, 0, (int)grid_size[1] * i);
				if(i == count_devices - 1 & count_devices * grid_size[1] != ((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1))) grid_size[1] +=1;
				grid_size[0] *= CUDABLOCK_SIZE;
				grid_size[1] *= CUDABLOCK_SIZE;
				std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};
				
				opencl::run_kernel_2d_timed(devices[i], svm_kernel_linear[i], grid_size, block_size);
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
		std::vector<real_t> ret(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		r_cl[0].from_device(ret); 
		std::copy(ret.begin(), ret.begin() + dept, r);
		for(int i = 1; i < count_devices; ++i){
			r_cl[i].from_device(ret);
			for(int i = 0; i < dept; ++i){
				r[i] += ret[i];
			}
		}
		std::vector<real_t> zeros(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 , 0); 

		for(int i = 0; i< dept; ++i) zeros[i] = r[i];
		for(int i = 0; i < count_devices; ++i) r_cl[i].to_device(zeros);
	}
	
	// for(int i = 0; i < dept; ++i){
	// 	std::cout << r[i] << "; ";
	// }
	std::cout << std::endl;
	
	real_t delta = mult(r, r, dept);	
	const real_t delta0 = delta;
	real_t alpha_cd, beta;
	real_t* Ad;


	std::vector<opencl::DevicePtrOpenCL<real_t> > Ad_cl;
	for(int i = 0; i < count_devices; ++i) Ad_cl.emplace_back(devices[i], dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1);
	Ad = new real_t[dept];

	

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		{
			std::vector<real_t> zeros( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
			for(int i = 0; i < count_devices; ++i) Ad_cl[i].to_device(zeros);
			//TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
			for(int i = 0; i < count_devices; ++i){
				std::vector<real_t> buffer( dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
				r_cl[i].from_device(buffer);
				for(int index = dept; index <  dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++index) buffer[index] = 0.0;
				r_cl[i].to_device(buffer);
			}
		}


		switch(kernel){
			case 0: 
				#pragma omp parallel for
				for(int i = 0; i < count_devices; ++i) {
					if (!svm_kernel_linear[i]) {
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						json::node &deviceNode =
							manager.get_configuration()["PLATFORMS"][devices[i].platformName]
													["DEVICES"][devices[i].deviceName];
						json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
						svm_kernel_linear[i] = manager.build_kernel(kernel_src, devices[i], kernelConfig, "kernel_linear");
					}
			
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);


					std::vector<size_t> grid_size{ ((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1)), (int)((dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) / count_devices)};
					//std::vector<size_t> grid_size{ ((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1)), (int)((dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1))};
					//TODO: bessere aufteilung (tuning parameter?)
					opencl::apply_arguments(svm_kernel_linear[i], q_cl[i].get(), Ad_cl[i].get(), r_cl[i].get(), data_cl[i].get(), QA_cost , 1/cost, Ncols, Nrows, 1, 0, (int) grid_size[1] * i);
					//opencl::apply_arguments(svm_kernel_linear[i], q_cl[i].get(), Ad_cl[i].get(), r_cl[i].get(), data_cl[i].get(), QA_cost , 1/cost, Ncols, Nrows, 1, 0, 0);
					if(i == count_devices - 1 & count_devices * grid_size[1] != ((int)(dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1))) grid_size[1] +=1;
					grid_size[0] *= CUDABLOCK_SIZE;
					grid_size[1] *= CUDABLOCK_SIZE;
					std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};
					
					opencl::run_kernel_2d_timed(devices[i], svm_kernel_linear[i], grid_size, block_size);
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

	



		// {
		// 	std::vector<real_t> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		// 	r_cl[0].from_device(buffer); 
		// 	std::copy(buffer.begin(), buffer.begin() + dept, d);
		// 	for(int i = 1; i < count_devices; ++i){
		// 		r_cl[i].from_device(buffer);
		// 		for(int i = 0; i < dept; ++i){
		// 			d[i] += buffer[i];
		// 		}
		// 	}
		// 	std::vector<real_t> zeros(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 , 0); 
		// 	for(int i = 0; i< dept; ++i) zeros[i] = d[i];
		// 	for(int i = 0; i < count_devices; ++i) r_cl[i].to_device(zeros);

		// }

			for(int i = 0; i< dept; ++i) d[i] = r[i];

		{
			std::vector<real_t> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
			std::vector<real_t> zeros(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 , 0); 
			Ad_cl[0].from_device(buffer); 
			std::copy(buffer.begin(), buffer.begin() + dept, Ad);
			std::copy(buffer.begin(), buffer.end() , zeros.begin());
			for(int i = 1; i < count_devices; ++i){
				Ad_cl[i].from_device(buffer);
				for(int i = 0; i < dept; ++i){
					Ad[i] += buffer[i];
				}
				for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i){
					zeros[i] += buffer[i];
				}
			} 
			for(int i = 0; i< dept; ++i) zeros[i] = Ad[i];
			for(int i = 0; i < count_devices; ++i) Ad_cl[i].to_device(zeros);

		}

		// {
		// 	std::vector<real_t> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		// 	// r_cl[CUDADEVICE].from_device(buffer);
		// 	// std::copy(buffer.begin(), buffer.begin()+dept, d);

		// 	Ad_cl[CUDADEVICE].from_device(buffer);
		// 	std::copy(buffer.begin(), buffer.begin()+dept, Ad);

		// }
		
		for(int i = 0; i < dept; ++i){
		std::cout << Ad[i] << "; ";
		}

		alpha_cd = delta / mult(d , Ad,  dept);
		
		//TODO: auf GPU
		std::vector<real_t> buffer_r(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		std::vector<real_t> buffer_x(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		x_cl[CUDADEVICE].from_device(buffer_x);
		r_cl[CUDADEVICE].from_device(buffer_r);
		add_mult_(((int) dept/1024) + 1, std::min(1024, dept),buffer_x.data(),buffer_r.data(),alpha_cd,dept);
		for(int i = 0; i < count_devices; ++i) x_cl[i].to_device(buffer_x);
		

		


		if(run%50 == 0){
			std::vector<real_t> buffer(b);
			for(int i = 0; i <  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++ i) buffer.push_back(0.0);
			for(int i = 0; i < count_devices; ++i) r_cl[i].to_device(buffer);

			switch(kernel){
			case 0: 
				#pragma omp parallel for
				for(int i = 0; i < count_devices; ++i){
					if (!svm_kernel_linear[i]) {
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						json::node &deviceNode =
							manager.get_configuration()["PLATFORMS"][devices[i].platformName]
													["DEVICES"][devices[i].deviceName];
						json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
						svm_kernel_linear[i] = manager.build_kernel(kernel_src, devices[i], kernelConfig, "kernel_linear");
					}
			
					
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD);
					opencl::apply_arguments(svm_kernel_linear[i], q_cl[i].get(), r_cl[i].get(), x_cl[i].get(), data_cl[i].get(), QA_cost , 1/cost, Ncols, Nrows, -1), 0, 0;
		
				std::vector<size_t> grid_size{((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE ,((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1) * CUDABLOCK_SIZE};
				std::vector<size_t> block_size{CUDABLOCK_SIZE, CUDABLOCK_SIZE};
				
				opencl::run_kernel_2d_timed(devices[i], svm_kernel_linear[i], grid_size, block_size);
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
	
			r_cl[CUDADEVICE].from_device(buffer);
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
			std::vector<real_t> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1, 0.0);
			std::copy(d, d+dept, buffer.begin());
			for(int i = 0; i < count_devices; ++i)r_cl[i].to_device(buffer);

		}
		
		
	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
	
	
	alpha.resize(dept);
	std::vector<real_t> ret_q(dept);


	{
		std::vector<real_t> buffer(dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 );
		x_cl[CUDADEVICE].from_device(buffer);
		std::copy(buffer.begin(), buffer.begin() + dept, alpha.begin());
		q_cl[CUDADEVICE].from_device(buffer);
		std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
	}
	delete[] d, Ad;
	return ret_q;
}