#include "CSVM.hpp"
#include "cuda-kernel.hpp"



#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>


#include "../src/OpenCL/manager/apply_arguments.hpp"
#include "../src/OpenCL/manager/run_kernel.hpp"

#include "distribution.hpp"



int count_devices = 1;


CSVM::CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){
	std::vector<opencl::device_t> &devices = manager.get_devices();
	first_device = devices[0];
	count_devices = devices.size();
	svm_kernel_linear.resize(count_devices, nullptr);
	kernel_q_cl.resize(count_devices, nullptr);
	std::cout << "GPUs found: " << count_devices << std::endl;
	
	}

void CSVM::loadDataDevice(){
	std::vector<opencl::device_t> &devices = manager.get_devices(); //TODO: header
	for(int device = 0; device < count_devices; ++device) datlast_cl.emplace_back(opencl::DevicePtrOpenCL<real_t> (devices[device], (Nfeatures_data)));
	std::vector<real_t> datalast(data[Ndatas_data - 1]);
	#pragma omp parallel
	for(int device = 0; device < count_devices; ++device) datlast_cl[device].to_device(datalast);
	#pragma omp parallel
	for(int device = 0; device < count_devices; ++device) datlast_cl[device].resize(Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);

	for(int device = 0; device < count_devices; ++device) data_cl.emplace_back(opencl::DevicePtrOpenCL<real_t>(devices[device], Nfeatures_data * (Ndatas_data - 1)));
	
	auto begin_transform = std::chrono::high_resolution_clock::now();
	const std::vector<real_t> transformet_data = transform_data(0, THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
	auto end_transform = std::chrono::high_resolution_clock::now();
	if(info){std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_transform-begin_transform).count() << " ms transformiert" << std::endl;}
	#pragma omp parallel
	for(int device = 0; device < count_devices; ++device){
		data_cl[device] = opencl::DevicePtrOpenCL<real_t>( devices[device], Nfeatures_data * (Ndatas_data - 1  + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) );
		data_cl[device].to_device(transformet_data);
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
	for(int device = 0; device < count_devices; ++device) x_cl.emplace_back( devices[device] , dept_all);
	for(int device = 0; device < count_devices; ++device) x_cl[device].to_device(x);
	
	std::vector<real_t> r(dept_all, 0.0);

	std::vector<opencl::DevicePtrOpenCL<real_t> > r_cl;
	for(int device = 0; device < count_devices; ++device)r_cl.emplace_back(devices[device], dept_all);
	

	std::vector<real_t> toDevice(dept_all, 0.0);
	std::copy(b.begin(), b.begin() + dept, toDevice.begin());
	r_cl[0].to_device(std::vector<real_t>(toDevice));
	#pragma omp parallel
	for(int device = 1; device < count_devices; ++device) r_cl[device].to_device(std::vector<real_t>(zeros));
	d = new real_t[dept];

	std::vector<opencl::DevicePtrOpenCL<real_t> > q_cl;
	for(int device = 0; device < count_devices; ++device) q_cl.emplace_back(devices[device], dept_all);
	//TODO: init on gpu
	for(int device = 0; device < count_devices; ++device) q_cl[device].to_device(std::vector<real_t>(dept_all, 0.0));
	std::cout << "kernel_q" << std::endl;
	#pragma omp parallel for
	for(int device = 0; device < count_devices; ++device){
		if (!kernel_q_cl[device]) {
			#pragma omp critical //TODO: evtl besser keine Referenz
			{
				std::string kernel_src_file_name{"../src/OpenCL/kernels/kernel_q.cl"};
				std::string kernel_src = manager.read_src_file(kernel_src_file_name);
				if (*typeid(real_t).name() == 'f') {
					manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
				}else if (*typeid(real_t).name() == 'd') {
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
		const int Ncols = Nfeatures_data;
		const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;

		q_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
		// resizeData(i,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
		const int start = device * Ncols / count_devices;
		const int end = (device + 1) * Ncols / count_devices;
	  	opencl::apply_arguments(kernel_q_cl[device], q_cl[device].get(), data_cl[device].get(), datlast_cl[device].get(), Nrows, start , end);
	
		size_t grid_size = static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE)) * THREADBLOCK_SIZE);
		size_t block_size = THREADBLOCK_SIZE ;
		opencl::run_kernel_1d_timed(devices[device], kernel_q_cl[device], grid_size, block_size);
	}
	{
		std::vector<real_t> buffer(dept_all);
		q_cl[0].from_device(buffer); //TODO:
		std::vector<real_t> ret(dept_all);
		for(int device = 1; device < count_devices; ++device){
			q_cl[device].from_device(ret);
			for(int i = 0; i < dept_all; ++i) buffer[i] += ret[i];
		} 
		#pragma omp parallel
		for(int device = 0; device < count_devices; ++device) q_cl[device].to_device(buffer);
	}

	switch(kernel){
		case 0: 
			#pragma omp parallel for
			for(int device = 0; device < count_devices; ++device){  
				if (!svm_kernel_linear[device]) {
					std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
					std::string kernel_src = manager.read_src_file(kernel_src_file_name);
					if (*typeid(real_t).name() == 'f') {
						manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
					}else if (*typeid(real_t).name() == 'd') {
						manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
					}
					json::node &deviceNode =
						manager.get_configuration()["PLATFORMS"][devices[device].platformName]
												["DEVICES"][devices[device].deviceName];
					json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];
					#pragma omp critical //TODO: evtl besser keine Referenz
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
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)))};
					const int start = device * Ncols / count_devices;
					const int end = (device + 1) * Ncols / count_devices;
					opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), r_cl[device].get(), x_cl[device].get(), data_cl[device].get(), QA_cost , 1/cost, Ncols, Nrows, -1, start, end);
					grid_size[0] *= THREADBLOCK_SIZE;
					grid_size[1] *= THREADBLOCK_SIZE;
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};

					opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
					// exit(0);
				}	
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
		r_cl[0].resize(dept_all);
		r_cl[0].from_device(r);
		for(int device = 1; device < count_devices; ++device){
			std::vector<real_t> ret(dept_all);
			r_cl[device].from_device(ret);
			for(int j = 0; j <= dept ; ++j){
				r[j] += ret[j];
			}
		}

		for(int device = 0; device < count_devices; ++device) r_cl[device].to_device(r);
	}
	real_t delta = mult(r.data(), r.data(), dept); //TODO:	
	const real_t delta0 = delta;
	real_t alpha_cd, beta;
	std::vector<real_t> Ad(dept);


	std::vector<opencl::DevicePtrOpenCL<real_t> > Ad_cl;
	for(int device = 0; device < count_devices; ++device) Ad_cl.emplace_back(devices[device], dept_all);

	

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		{
			#pragma omp parallel
			for(int device = 0; device < count_devices; ++device) Ad_cl[device].to_device(zeros);
			//TODO: effizienter auf der GPU implementieren (evtl clEnqueueFillBuffer )
			#pragma omp parallel
			for(int device = 0; device < count_devices; ++device) {
				std::vector<real_t> buffer( dept_all);
				r_cl[device].resize(dept_all);
				r_cl[device].from_device(buffer);
				for(int index = dept; index < dept_all; ++index) buffer[index] = 0.0;
				r_cl[device].to_device(buffer);
			}
		}
		switch(kernel){
			case 0: 
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					if (!svm_kernel_linear[device]){
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						if (*typeid(real_t).name() == 'f') {
							manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
						}else if (*typeid(real_t).name() == 'd') {
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
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)))};
					const int start = device * Ncols / count_devices;
					const int end = (device + 1) * Ncols / count_devices;
					opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), Ad_cl[device].get(), r_cl[device].get(), data_cl[device].get(), QA_cost , 1/cost, Ncols, Nrows, 1, start, end);
					grid_size[0] *= THREADBLOCK_SIZE;
					grid_size[1] *= THREADBLOCK_SIZE;
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};

					opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
					}
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

		{ 
			std::vector<real_t> buffer(dept_all, 0);
			for(int device = 0; device < count_devices; ++device){
			// for(int device = 0; device < 1; ++device){
				std::vector<real_t> ret(dept_all, 0 );
				Ad_cl[device].resize(dept_all);
				Ad_cl[device].from_device(ret);
				for(int j = 0; j < dept  ; ++j){
						buffer[j] += ret[j];
					}
			}
			std::copy(buffer.begin(), buffer.begin()+dept, Ad.data());
			for(int device = 0; device < count_devices; ++device) Ad_cl[device].to_device(buffer);
		}
	

		alpha_cd = delta / mult(d , Ad.data(),  dept);
		//TODO: auf GPU
		std::vector<real_t> buffer_r(dept_all );
		r_cl[0].resize(dept_all);
		r_cl[0].from_device(buffer_r);
		add_mult_(((int) dept/1024) + 1, std::min(1024, (int) dept),x.data(),buffer_r.data(),alpha_cd,dept);
		#pragma omp parallel
		for(int device = 0; device < count_devices; ++device) x_cl[device].resize(dept_all);
		#pragma omp parallel
		for(int device = 0; device < count_devices; ++device) x_cl[device].to_device(x);

		

		

		if(run%50 == 49){
			std::vector<real_t> buffer(b);
			buffer.resize(dept_all);
			r_cl.resize(dept_all);
			r_cl[0].to_device(buffer);
			#pragma omp parallel
			for(int device = 1; device < count_devices; ++device) r_cl[device].to_device(zeros);
			switch(kernel){
			case 0: 
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					if (!svm_kernel_linear[device]) {
						std::string kernel_src_file_name{"../src/OpenCL/kernels/svm-kernel-linear.cl"};
						std::string kernel_src = manager.read_src_file(kernel_src_file_name);
						if (*typeid(real_t).name() == 'f') {
							manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
						}else if (*typeid(real_t).name() == 'd') {
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
					r_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					x_cl[device].resize(dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE); 
					// resizeData(device,THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					const int start = device * Ncols / count_devices;
					const int end = (device + 1) * Ncols / count_devices;
					std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE)))};
					opencl::apply_arguments(svm_kernel_linear[device], q_cl[device].get(), r_cl[device].get(), x_cl[device].get(), data_cl[device].get(), QA_cost, 1/cost, Ncols, Nrows, -1, start, end);
					grid_size[0] *= THREADBLOCK_SIZE;
					grid_size[1] *= THREADBLOCK_SIZE;
					std::vector<size_t> block_size{THREADBLOCK_SIZE, THREADBLOCK_SIZE};
					
					opencl::run_kernel_2d_timed(devices[device], svm_kernel_linear[device], grid_size, block_size);
					}
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
				r_cl[0].resize(dept_all);
				r_cl[0].from_device(r);
				for(int device = 1; device < count_devices; ++device){
					std::vector<real_t> ret(dept_all, 0 );
					r_cl[device].resize(dept_all);
					r_cl[device].from_device(ret);
					for(int j = 0; j <= dept ; ++j){
						r[j] += ret[j];
					}
				}
				#pragma omp parallel
				for(int device = 0; device < count_devices; ++device) r_cl[device].to_device(r);
			}
		}else{
			for(int index = 0; index < dept; ++index){
				r[index] -= alpha_cd * Ad[index];
			}
		}
		
		delta = mult(r.data() , r.data(), dept); //TODO:
		if(delta < eps * eps * delta0) break;
		beta = -mult(r.data(), Ad.data(), dept) / mult(d, Ad.data(), dept); //TODO:
		add(mult(beta, d, dept),r.data(), d, dept);//TODO:


		{
			std::vector<real_t> buffer(dept_all, 0.0);
			std::copy(d, d+dept, buffer.begin());
			#pragma omp parallel
			for(int device = 0; device < count_devices; ++device) r_cl[device].resize(dept_all);
			#pragma omp parallel
			for(int device = 0; device < count_devices; ++device) r_cl[device].to_device(buffer);

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
	delete[] d;
	return ret_q;
}