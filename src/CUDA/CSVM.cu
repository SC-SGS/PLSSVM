#include "CSVM.hpp"
#include "cuda-kernel.hpp"
#include "cuda-kernel.cuh"
#include "svm-kernel.cuh"
#include "templates.hpp"

int count_devices = 1;

CSVM::CSVM(const real_t cost, const real_t epsilon, const unsigned kernel, const real_t degree, const real_t gamma, const real_t coef0 , const bool info) : 
		cost(cost), epsilon(epsilon), kernel(kernel), degree(degree), gamma(gamma), coef0(coef0), info(info){
	gpuErrchk(cudaGetDeviceCount(&count_devices));
	datlast_d = std::vector<real_t *>(count_devices);
	data_d = std::vector<real_t *>(count_devices);
	std::clog << "GPUs found: " << count_devices << std::endl;	
}


void CSVM::loadDataDevice(){
	
	for(int device = 0; device < count_devices; ++device){ 
		cuda_set_device(device);
		cuda_malloc(datlast_d[device], Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
	}
	std::vector<real_t> datalast(data[Ndatas_data - 1]);
	datalast.resize(Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
	#pragma omp parallel for
	for(int device = 0; device < count_devices; ++device) { 
		cuda_set_device(device);
		cuda_memcpy(datlast_d[device], datalast, (Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE), cudaMemcpyHostToDevice);
	}
	datalast.resize(Ndatas_data - 1);
	for(int device = 0; device < count_devices; ++device) {
		cuda_set_device(device);
		cuda_malloc(data_d[device], Nfeatures_data * (Ndatas_data + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE));
	}
	auto begin_transform = std::chrono::high_resolution_clock::now();
	const std::vector<real_t> transformed_data = transform_data(0, THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
	auto end_transform = std::chrono::high_resolution_clock::now();
	if(info){
		std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_transform-begin_transform).count() << " ms transformiert" << std::endl;
		}
	#pragma omp parallel for
	for(int device = 0; device < count_devices; ++device){
		cuda_set_device(device);
		cuda_memcpy(data_d[device], transformed_data, Nfeatures_data * (Ndatas_data - 1  + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE), cudaMemcpyHostToDevice);
	}
}


std::vector<real_t>CSVM::CG(const std::vector<real_t> &b,const int imax,  const real_t eps)
{
	const unsigned dept = Ndatas_data - 1;
	const unsigned boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
	const unsigned dept_all = dept + boundary_size;

	// dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);
	
	real_t* d_d;
	cuda_set_device(); 
	cuda_malloc(d_d, dept_all);
	std::vector<real_t> x(dept_all , 1.0);
	std::fill(x.end() - boundary_size, x.end(), 0.0);
	

	std::vector<real_t *> x_d(count_devices);
	std::vector<real_t> r(dept_all, 0.0);
	std::vector<real_t *> r_d(count_devices);

	#pragma omp parallel for
	for(int device = 0; device < count_devices; ++device) { 
		cuda_set_device(device);
		cuda_malloc(x_d[device], dept_all);
		cuda_memcpy(x_d[device], x, dept_all, cudaMemcpyHostToDevice);
		cuda_malloc(r_d[device], dept_all);
	}

	cuda_set_device();
	cuda_memcpy(r_d[0], b, dept, cudaMemcpyHostToDevice);
	cuda_memset(r_d[0] + dept, 0, boundary_size);
	#pragma omp parallel for
	for(int device = 1; device < count_devices; ++device) { 
		cuda_set_device(device);
		cuda_memset(r_d[device] , 0, dept_all);
	}

	std::vector<real_t *> q_d(count_devices);
	for(int device = 0; device < count_devices; ++device) { 
		cuda_set_device(device);
		cuda_malloc(q_d[device], dept_all);
		cuda_memset(q_d[device] , 0, dept_all);
	}
	if(info)std::clog << "kernel_q" << std::endl;

	cuda_device_synchronize();
	for(int device = 0; device < count_devices; ++device){
		cuda_set_device(device);
		const int Ncols = Nfeatures_data;
		const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
		
		const int start = device * Ncols / count_devices;
		const int end = (device + 1) * Ncols / count_devices;
		kernel_q<<<(dept/CUDABLOCK_SIZE) + 1, std::min(CUDABLOCK_SIZE, dept)>>>(q_d[device], data_d[device], datlast_d[device], Nrows, start, end );
		gpuErrchk(cudaPeekAtLastError());
	}
	cuda_set_device();
	cuda_device_synchronize();
	real_t* buffer_d;
	cuda_malloc(buffer_d, dept_all);
	for(int device = 1; device < count_devices; ++device){
		cuda_device_synchronize();
		cuda_memcpy(buffer_d, 0, q_d[device], device, dept);
		add_inplace<<< ( dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(q_d[0],buffer_d,dept);
	}
	gpuErrchk(cudaPeekAtLastError());
	cuda_device_synchronize();
	for(int device = 1; device < count_devices; ++device){
		cuda_memcpy(q_d[device], device, q_d[0], 0, dept);
	}


	switch(kernel){
		case 0: 
			{
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					cuda_set_device(device);
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					dim3 grid(static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
					const int start = device * Ncols / count_devices;
					const int end = (device + 1) * Ncols / count_devices;
					kernel_linear<<<grid,block>>>( q_d[device], r_d[device], x_d[device], data_d[device], QA_cost, 1/cost, Ncols , Nrows, -1, start, end);
					gpuErrchk( cudaPeekAtLastError() );
				}
				break;
			}
		case 1: 
			// kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
			break;	
		case 2: 
			// kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
			break;
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	real_t delta, *delta_d;
	
	cuda_set_device();
	for(int device = 1; device < count_devices; ++device){
		cuda_device_synchronize();
		cuda_memcpy(buffer_d, 0, r_d[device], device, dept);
		add_inplace<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],buffer_d,dept);
	}
	cuda_device_synchronize();
	for(int device = 1; device < count_devices; ++device){
		cuda_memcpy(r_d[device], device, r_d[0], 0, dept);
	}	
	
	cuda_malloc(delta_d,1);
	cuda_memset(delta_d, 0, 1);
	dot<<< ( dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],r_d[0],delta_d,dept);
	cuda_device_synchronize();
	cuda_memcpy(delta, delta_d, cudaMemcpyDeviceToHost);

	// real_t delta = mult(r.data(), r.data(), dept);

	const real_t delta0 = delta;
	
	real_t alpha_cd, beta;
	std::vector<real_t *> Ad_d(count_devices);
	for(int device = 0; device < count_devices; ++device) { 
		cuda_set_device(device);
		cuda_malloc(Ad_d[device], dept_all);
	}

	real_t *ret_d;
	cuda_set_device();
	cuda_malloc(ret_d, 1);


	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::clog << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		for(int device = 0; device < count_devices; ++device) { 
			cuda_set_device(device);
			cuda_memset(Ad_d[device], 0, dept_all);
			cuda_memset(r_d[device] + dept, 0, boundary_size);
		}	
		switch(kernel){
			case 0:
			{
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					cuda_set_device(device);
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					dim3 grid(static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
					const int start = device * Ncols / count_devices;
					const int end = (device + 1) * Ncols / count_devices;
					kernel_linear<<<grid,block>>>( q_d[device], Ad_d[device], r_d[device], data_d[device], QA_cost, 1/cost, Ncols , Nrows, 1, start, end);
					gpuErrchk(cudaPeekAtLastError());
				}
			}
				break;
			case 1: 
				// kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
				break;
			case 2: 
				// kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
				break;
			default: throw std::runtime_error("Can not decide wich kernel!");
		}
		cuda_set_device();
		cuda_memcpy(d_d, r_d[0], dept, cudaMemcpyDeviceToDevice);
		cuda_device_synchronize();

		
		for(int device = 1; device < count_devices; ++device){
			cuda_device_synchronize();
			cuda_memcpy(buffer_d, 0, Ad_d[device], device, dept);
			add_inplace<<< ((int) dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, (unsigned) dept)>>>(Ad_d[0],buffer_d,dept);
		}
		cuda_memset(Ad_d[0]+dept,0,boundary_size);
		cuda_device_synchronize();
		for(int device = 1; device < count_devices; ++device){
			cuda_memcpy(Ad_d[device], device, Ad_d[0], 0, dept_all);
		}
		cuda_set_device();
		real_t ret = 0;
		
		cuda_memset(ret_d, 0, 1);
		dot<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(d_d,Ad_d[0],ret_d,dept);
		cuda_device_synchronize();
		cuda_memcpy(&ret, ret_d, 1, cudaMemcpyDeviceToHost);
		
		alpha_cd = delta / ret;

		
		add_mult<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(x_d[0],r_d[0],alpha_cd,dept);
		cuda_device_synchronize();
		for(int device = 1; device < count_devices; ++device){
			cuda_memcpy(x_d[device], device, x_d[0], 0, dept);
		}

		if(run%50 == 49){
			std::vector<real_t> buffer(b);
			buffer.resize(dept_all);
			cuda_set_device();
			cuda_memcpy(r_d[0], buffer, dept_all , cudaMemcpyHostToDevice);
			#pragma omp parallel for
			for(int device = 1; device < count_devices; ++device) { 
				cuda_set_device(device);
				cuda_memset(r_d[device], 0, dept_all);
			}
			switch(kernel){
				case 0: 
					{
						#pragma omp parallel for
						for(int device = 0; device < count_devices; ++device){
							cuda_set_device(device);
							const int Ncols = Nfeatures_data;
							const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
							const int start = device * Ncols / count_devices;
							const int end = (device + 1) * Ncols / count_devices;
							dim3 grid(static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<unsigned>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
							kernel_linear<<<grid,block>>>( q_d[device], r_d[device], x_d[device], data_d[device], QA_cost, 1/cost, Ncols, Nrows, -1, start, end);
							gpuErrchk(cudaPeekAtLastError());
						}
					}
					break;
				case 1: 
					// kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
					break;
				case 2: 
					// kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
					break;
				default: throw std::runtime_error("Can not decide wich kernel!");
			}
			cuda_device_synchronize();	
			cuda_set_device();
			for(int device = 1; device < count_devices; ++device){
				cuda_device_synchronize();
				cuda_memcpy(buffer_d, 0, r_d[device], device, dept);
				add_inplace<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],buffer_d,dept);
			}
			cuda_device_synchronize();
			for(int device = 1; device < count_devices; ++device){
				cuda_memcpy(r_d[device], device, r_d[0], 0, dept);
			}


		}else{
			// for(int index = 0; index < dept; ++index){
			// 	r[index] -= alpha_cd * Ad[index];
			// }
			alpha_cd *= -1;
			add_mult<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],Ad_d[0],alpha_cd,dept);
			cuda_device_synchronize();

		}

		// delta = mult(r.data() , r.data(), dept); 
		cuda_set_device();
		cuda_memset(delta_d, 0, 1);
		dot<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],r_d[0],delta_d,dept);
		cuda_device_synchronize();
		cuda_memcpy(&delta, delta_d, 1, cudaMemcpyDeviceToHost);
		
		if(delta < eps * eps * delta0) break;
		
		//beta = -mult(r.data(), Ad.data(), dept) / mult(d, Ad.data(), dept); //TODO: multigpu
		cuda_set_device();
		cuda_memset(ret_d, 0, 1);
		dot<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],Ad_d[0],ret_d,dept);
		cuda_device_synchronize();
		cuda_memcpy(&ret, ret_d, 1, cudaMemcpyDeviceToHost);
		beta = -ret;
		cuda_memset(ret_d, 0, 1);
		dot<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(buffer_d,Ad_d[0],ret_d,dept);
		cuda_device_synchronize();
		cuda_memcpy(&ret, ret_d, 1, cudaMemcpyDeviceToHost);
		beta /= ret;
		

		
		// add(mult(beta, d, dept),r.data(), d, dept);
		add_mult<<< (dept/THREADS_PER_BLOCK) + 1, std::min(THREADS_PER_BLOCK, dept)>>>(r_d[0],d_d,beta,dept);
		cuda_device_synchronize();
		cuda_memset(r_d[0] + dept, 0, boundary_size);
		
		#pragma omp parallel for
		for(int device = 1; device < count_devices; ++device) { 
			cuda_memcpy(r_d[device], device, r_d[0], 0, dept_all);
		}
	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;

	gpuErrchk(cudaFree(delta_d));
	gpuErrchk(cudaFree(ret_d));
	alpha.resize(dept);
	std::vector<real_t> ret_q(dept);
	cuda_set_device();
	cuda_memcpy(alpha,x_d[0], dept, cudaMemcpyDeviceToHost);
	cuda_memcpy(ret_q, q_d[0], dept, cudaMemcpyDeviceToHost);

	return ret_q;
}