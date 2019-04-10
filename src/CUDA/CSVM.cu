#include "CSVM.hpp"
#include "cuda-kernel.hpp"
#include "cuda-kernel.cuh"
#include "svm-kernel.cuh"

int CUDADEVICE = 0;







#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}






int count_devices = 1;


CSVM::CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){
	cudaGetDeviceCount(&count_devices);
	datlast_d = std::vector<real_t *>(count_devices);
	data_d = std::vector<real_t *>(count_devices);



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
	for(int device = 0; device < count_devices; ++device){ cudaSetDevice(device);gpuErrchk(cudaMalloc((void **) &datlast_d[device], (Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_t)));}
	std::vector<real_t> datalast(data[Ndatas_data - 1]);
	datalast.resize(Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE);
	#pragma opm parallel
	for(int device = 0; device < count_devices; ++device) { cudaSetDevice(device); gpuErrchk(cudaMemcpy(datlast_d[device], datalast.data(), (Ndatas_data - 1 + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_t),cudaMemcpyHostToDevice));}
	datalast.resize(Ndatas_data - 1);
	for(int device = 0; device < count_devices; ++device) {cudaSetDevice(device); gpuErrchk(cudaMalloc((void **) &data_d[device], Nfeatures_data * (Ndatas_data + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE) * sizeof(real_t))); }

	
	#pragma opm parallel
	for(int device = 0; device < count_devices; ++device){
		loadDataDevice(device, THREADBLOCK_SIZE * INTERNALBLOCK_SIZE, 0 , Ndatas_data - 1);
	}
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
    if(info){std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << " ms geschrieben" << std::endl;
    }else if(times){
		std::clog << data.size()<<", "<< Nfeatures_data  <<", " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << ", "<< std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - end_parse).count()<< ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_gpu).count() << ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << std::endl;
	} 

}



void CSVM::loadDataDevice(const int device, const int boundary, const int start_line, const int number_lines){
	cudaSetDevice(device);

	std::vector<real_t> vec;
	//vec.reserve(Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1);
	for(size_t col = 0; col < Nfeatures_data; ++col){
		for(size_t row = start_line; row < Ndatas_data - 1; ++row){
			vec.push_back(data[row][col]);
		}
		for(int i = 0 ; i < boundary ; ++i){
			vec.push_back(0.0);
		}
	}
	gpuErrchk(cudaMemcpy(data_d[device], vec.data(), Nfeatures_data * (Ndatas_data - start_line + boundary) * sizeof(real_t), cudaMemcpyHostToDevice));
}




std::vector<real_t>CSVM::CG(const std::vector<real_t> &b,const int imax,  const real_t eps)
{
	const size_t dept = Ndatas_data - 1;
	const size_t boundary_size = THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
	const size_t dept_all = dept + boundary_size;
	std::vector<real_t> zeros(dept_all, 0.0);

	// dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	dim3 block(THREADBLOCK_SIZE, THREADBLOCK_SIZE);
	
	real_t *d;
	std::vector<real_t> x(dept_all , 1.0);
	std::fill(x.end() - boundary_size, x.end(), 0.0);
	

	std::vector<real_t *> x_d(count_devices);
	std::vector<real_t> r(dept_all, 0.0);
	std::vector<real_t *> r_d(count_devices);
	for(int device = 0; device < count_devices; ++device) { 
		cudaSetDevice(device); 
		gpuErrchk(cudaMalloc((void **) &x_d[device], dept_all * sizeof(real_t)));
		gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all *sizeof(real_t), cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc((void **) &r_d[device], dept_all * sizeof(real_t)));
	}
	
	

	
	
	gpuErrchk(cudaSetDevice(0));
	gpuErrchk(cudaMemcpy(r_d[0], b.data(), dept * sizeof(real_t), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(r_d[0] + dept, 0, (dept_all - dept) * sizeof(real_t)));
	#pragma opm parallel
	for(int device = 1; device < count_devices; ++device) { cudaSetDevice(device); gpuErrchk(cudaMemset(r_d[device] , 0, dept_all * sizeof(real_t)));}
	d = new real_t[dept];

	std::vector<real_t *> q_d(count_devices);
	for(int device = 0; device < count_devices; ++device) { 
		cudaSetDevice(device); gpuErrchk(cudaMalloc((void **) &q_d[device], dept_all * sizeof(real_t)));
		gpuErrchk(cudaMemset(q_d[device] , 0, dept_all * sizeof(real_t)));
	}
	std::cout << "kernel_q" << std::endl;

	gpuErrchk(cudaDeviceSynchronize());
	for(int device = 0; device < count_devices; ++device){
		cudaSetDevice(device);
		const int Ncols = Nfeatures_data;
		const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
		
		const int start = device * Ncols / count_devices;
		const int end = (device + 1) * Ncols / count_devices;
		kernel_q<<<((int) dept/CUDABLOCK_SIZE) + 1, std::min((size_t)CUDABLOCK_SIZE, dept)>>>(q_d[device], data_d[device], datlast_d[device], Nrows, start, end );
		gpuErrchk( cudaPeekAtLastError() );
	}
	gpuErrchk(cudaDeviceSynchronize());
	{
		std::vector<real_t> buffer(dept_all);
		gpuErrchk(cudaSetDevice(0));
		gpuErrchk(cudaMemcpy(buffer.data(), q_d[0],  dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
		std::vector<real_t> ret(dept_all);
		for(int device = 1; device < count_devices; ++device){
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaMemcpy(ret.data(), q_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
			for(int i = 0; i < dept_all; ++i) buffer[i] += ret[i];
		} 
		
		#pragma opm parallel
		for(int device = 0; device < count_devices; ++device) { 
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaMemcpy(q_d[device], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
		}
	}


	switch(kernel){
		case 0: 
			{
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					gpuErrchk(cudaSetDevice(device));
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					dim3 grid(static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
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
 
	// cudaMemcpy(r, r_d, dept*sizeof(real_t), cudaMemcpyDeviceToHost);
	gpuErrchk(cudaDeviceSynchronize());
	{
		gpuErrchk(cudaSetDevice(0));
		gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
		for(int device = 1; device < count_devices; ++device){
			gpuErrchk(cudaSetDevice(device));
			std::vector<real_t> ret(dept_all);
			gpuErrchk(cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
			for(int j = 0; j <= dept ; ++j){
				r[j] += ret[j];
			}
		}
	}
	real_t delta = mult(r.data(), r.data(), dept); //TODO:	
	const real_t delta0 = delta;
	real_t alpha_cd, beta;
	std::vector<real_t> Ad(dept);


	std::vector<real_t *> Ad_d(count_devices);
	for(int device = 0; device < count_devices; ++device) { 
		gpuErrchk(cudaSetDevice(device));
		gpuErrchk(cudaMalloc((void **) &Ad_d[device], dept_all  *sizeof(real_t)));
		gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
	}
	//cudaMallocHost((void **) &Ad, dept *sizeof(real_t));



	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		for(int device = 0; device < count_devices; ++device) { 
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaMemset(Ad_d[device], 0, dept_all * sizeof(real_t)));
			gpuErrchk(cudaMemset(r_d[device] + dept, 0, (dept_all - dept) * sizeof(real_t)));
		}	
		switch(kernel){
			case 0:
			{
				#pragma omp parallel for
				for(int device = 0; device < count_devices; ++device){
					gpuErrchk(cudaSetDevice(device));
					const int Ncols = Nfeatures_data;
					const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
					dim3 grid(static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
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


		for(int i = 0; i< dept; ++i) d[i] = r[i];
		
		gpuErrchk(cudaDeviceSynchronize());
	{
		std::vector<real_t> buffer(dept_all, 0);
		for(int device = 0; device < count_devices; ++device){
			gpuErrchk(cudaSetDevice(device));
			std::vector<real_t> ret(dept_all, 0 );
			gpuErrchk(cudaMemcpy(ret.data(), Ad_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
			for(int j = 0; j <= dept ; ++j){
				buffer[j] += ret[j];
			}
		}
		std::copy(buffer.begin(), buffer.begin()+dept, Ad.data());
		for(int device = 0; device < count_devices; ++device) { 
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaMemcpy(Ad_d[device], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
		}
	}

		alpha_cd = delta / mult(d , Ad.data(),  dept);
		// add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
		//TODO: auf GPU
		std::vector<real_t> buffer_r(dept_all );
		cudaSetDevice(0);
		gpuErrchk(cudaMemcpy(buffer_r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
		add_mult_(((int) dept/1024) + 1, std::min(1024, (int) dept),x.data(),buffer_r.data(),alpha_cd,dept);

		#pragma opm parallel
		for(int device = 0; device < count_devices; ++device) { 
			gpuErrchk(cudaSetDevice(device));
			gpuErrchk(cudaMemcpy(x_d[device], x.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
		}
		if(run%50 == 49){
			std::vector<real_t> buffer(b);
			buffer.resize(dept_all);
			cudaSetDevice(0);
			gpuErrchk(cudaMemcpy(r_d[0], buffer.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
			#pragma opm parallel
			for(int device = 1; device < count_devices; ++device) { 
				gpuErrchk(cudaSetDevice(device));
				gpuErrchk(cudaMemset(r_d[device], 0, dept_all * sizeof(real_t)));
			}
			switch(kernel){
				case 0: 
					{
						#pragma omp parallel for
						for(int device = 0; device < count_devices; ++device){
							gpuErrchk(cudaSetDevice(device));
							const int Ncols = Nfeatures_data;
							const int Nrows = dept + THREADBLOCK_SIZE * INTERNALBLOCK_SIZE;
							const int start = device * Ncols / count_devices;
							const int end = (device + 1) * Ncols / count_devices;
							dim3 grid(static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))),static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(THREADBLOCK_SIZE * INTERNALBLOCK_SIZE))));
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
			gpuErrchk(cudaDeviceSynchronize());	
			// cudaMemcpy(r, r_d, dept*sizeof(real_t), cudaMemcpyDeviceToHost);

			{
				gpuErrchk(cudaSetDevice(0));
				gpuErrchk(cudaMemcpy(r.data(), r_d[0], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
				#pragma opm parallel
				for(int device = 1; device < count_devices; ++device){
					gpuErrchk(cudaSetDevice(device));
					std::vector<real_t> ret(dept_all, 0 );
					gpuErrchk(cudaMemcpy(ret.data(), r_d[device], dept_all * sizeof(real_t), cudaMemcpyDeviceToHost));
					for(int j = 0; j <= dept ; ++j){
						r[j] += ret[j];
					}
				}
				#pragma opm parallel
				for(int device = 0; device < count_devices; ++device) { 
					gpuErrchk(cudaSetDevice(device));
					gpuErrchk(cudaMemcpy(r_d[device], r.data(), dept_all * sizeof(real_t), cudaMemcpyHostToDevice));
				}
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
			#pragma opm parallel
			for(int device = 0; device < count_devices; ++device) { 
				gpuErrchk(cudaSetDevice(device)); 
				gpuErrchk(cudaMemcpy(r_d[device], buffer.data(), dept_all*sizeof(real_t), cudaMemcpyHostToDevice));
			}

		}


	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;


	alpha.resize(dept);
	std::vector<real_t> ret_q(dept);
	gpuErrchk(cudaDeviceSynchronize());
	{
		std::vector<real_t> buffer(dept_all );
		std::copy(x.begin(), x.begin() + dept, alpha.begin());
		gpuErrchk(cudaSetDevice(0)); 
		gpuErrchk(cudaMemcpy(buffer.data(), q_d[0], dept_all*sizeof(real_t), cudaMemcpyDeviceToHost));
		std::copy(buffer.begin(), buffer.begin() + dept, ret_q.begin());
	}
	// cudaMemcpy(&alpha[0],x_d, dept * sizeof(real_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(&ret_q[0],q_d, dept * sizeof(real_t), cudaMemcpyDeviceToHost);
	// cudaFree(Ad_d);
	// cudaFree(r_d);
	// cudaFree(datlast);
	// cudaFreeHost(Ad);
	// cudaFree(x_d);
	// cudaFreeHost(r);
	// cudaFreeHost(d);
	return ret_q;
}