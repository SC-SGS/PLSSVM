#include "CSVM.hpp"
///#include "cuda-kernel.cuh"
#include "svm-kernel.cuh"
//#include "svm-kernel.hpp"
#include "cuda-kernel.hpp"


#ifdef WITH_OPENCL
#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "../src/OpenCL/DevicePtrOpenCL.hpp"
#include <stdexcept>
#endif

int CUDADEVICE = 0;

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){

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
	dim3 grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	//std::tuple<int,int> grid((int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1,(int)dept/(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) + 1);
	dim3 block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
	//std::tuple<int,int> block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
	double *x_d, *r, *d, *r_d, *q_d;

	//cudaMalloc((void **) &x_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)*sizeof(double));
	x_d = new double[dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1];
	//init<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,1,dept);
	init(((int) dept/1024) + 1, std::min(1024, dept),x_d,1,dept);
	//init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1>>>(x_d + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	init(1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1,x_d + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);

	//r = b - (A * x)
	///r = b;

	//cudaMallocHost((void **) &r, dept *sizeof(double));
	r = new double[dept];

	//cudaMalloc((void **) &r_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) *sizeof(double));
	r_d = new double[dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1];

	// init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD)>>>(r_d + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	init( 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), r_d + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	//cudaMallocHost((void **) &d, dept *sizeof(double));
	d = new double[dept];
	//cudaMemcpy(r_d,&b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
	for(int i = 0; i < dept; ++i) r_d[i] = b[i];
	
	//cudaMalloc((void **) &q_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
	q_d = new double[dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1];

	//cudaMemset(q_d, 0, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  * sizeof(double));
	for(int i = 0; i < (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1); ++i) q_d[i] = 0;
	//kernel_q<<<((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept)>>>(q_d, data_d, datlast, Nfeatures_data , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
	kernel_q(((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept),q_d, data_d, datlast, Nfeatures_data , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
	// std::cout << "q_d: " ;
	// for(int i = 0; i < dept; ++i) std::cout << q_d[i] << ", ";
	// std::cout  << std::endl;
	double *q_d_d;
	cudaMalloc((void **) &q_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
	cudaMemcpy(q_d_d, q_d, (dept ) * sizeof(double), cudaMemcpyHostToDevice);
	double *r_d_d;
	cudaMalloc((void **) &r_d_d, (dept) * sizeof(double));
	cudaMemcpy(r_d_d, r_d, (dept) * sizeof(double), cudaMemcpyHostToDevice);
	double *x_d_d;
	cudaMalloc((void **) &x_d_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)*sizeof(double));
	cudaMemcpy(x_d_d, x_d, (dept ) * sizeof(double), cudaMemcpyHostToDevice);
	double *data_d_d;
	cudaMalloc((void **) &data_d_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double));
	cudaMemcpy(data_d_d, data_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	switch(kernel){
		case 0: 
			kernel_linear<<<grid,block>>>(q_d_d, r_d_d, x_d_d ,data_d_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
			
			//std::cout << "Error: " << cudaGetLastError() << std::endl;
    		//printf( "Error!\n" );
			//kernel_linear(grid,block,q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
			//void kernel_linear(const std::vector<double> &b, std::vector<std::vector<double>> &data, double *datlast, double *ret, const double *d, const int dim,const double QA_cost, const double cost, const int add){
			//kernel_linear(b, data, datlast,q_d, r_d, x_d, dept, QA_cost, 1/cost,-1);
			break;
		case 1: 
			//kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
			std::cerr << "Wird noch nicht unterstützt" << std::endl;
			break;	
		case 2: 
			//kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
			std::cerr << "Wird noch nicht unterstützt" << std::endl;
			break;
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	
	cudaDeviceSynchronize();
	cudaMemcpy(q_d, q_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(r_d, r_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(x_d, x_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(q_d_d);
	cudaFree(r_d_d);
	cudaFree(x_d_d);
	cudaFree(data_d_d);

	std::cout << "QA_cost:" << QA_cost << " 1/cost:" << 1/cost << " Nfeatures_data:" << Nfeatures_data << " dept+:" <<  dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) << std::endl;
	//cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i) r[i] = r_d[i];
	std::cout << "x_d: ";
	for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i) std::cout << x_d[i] << ", ";
	std::cout  << std::endl;std::cout << std::endl;
	std::cout << "r_d: ";
	for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i) std::cout << r_d[i] << ", ";
	std::cout  << std::endl;std::cout << std::endl;


	std::cout << "q_d: ";
	for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i) std::cout << q_d[i] << ", ";
	std::cout  << std::endl;std::cout << std::endl;
	std::cout << "data_d: ";
	for(int i = 0; i < dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1; ++i) std::cout << data_d[i] << ", ";
	std::cout  << std::endl;std::cout << std::endl;


	double delta = mult(r, r, dept);	
	const double delta0 = delta;
	double alpha_cd, beta;
	double* Ad, *Ad_d;
	//cudaMallocHost((void **) &Ad, dept *sizeof(double));
	Ad = new double[dept];
	//cudaMalloc((void **) &Ad_d, (dept +(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  *sizeof(double));
	Ad_d = new double[dept +(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1];

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		//cudaMemset(Ad_d, 0, dept * sizeof(double));
		for(int i = 0; i < dept; ++i) Ad_d[i] = 0;
		//cudaMemset(r_d + dept, 0, ((CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
		for(int i = dept; i < (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1 + dept; ++i) r_d[i] = 0;


		// double *q_d_d;
		cudaMalloc((void **) &q_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
		cudaMemcpy(q_d_d, q_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyHostToDevice);
		// double *r_d_d;
		cudaMalloc((void **) &r_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
		cudaMemcpy(r_d_d, r_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyHostToDevice);
		// double *x_d_d;
		cudaMalloc((void **) &x_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
		cudaMemcpy(x_d_d, x_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyHostToDevice);
		// double *data_d_d;
		cudaMalloc((void **) &data_d_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double));
		cudaMemcpy(data_d_d, x_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double), cudaMemcpyHostToDevice);
	
		switch(kernel){
			case 0: 
				kernel_linear<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1);
				//kernel_linear(grid,block,q_d, Ad_d, r_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
				//kernel_linear(b, data, datlast, q_d, Ad_d, r_d, data.back().size(), QA_cost, 1/cost,-1);
				break;
			case 1: 
				//kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
				std::cerr << "Wird noch nicht unterstützt" << std::endl;
				break;
			case 2: 
				//kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
				std::cerr << "Wird noch nicht unterstützt" << std::endl;
				break;
			default: throw std::runtime_error("Can not decide wich kernel!");
		}



		cudaDeviceSynchronize();
		cudaMemcpy(q_d, q_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(r_d, r_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(x_d, x_d_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaFree(q_d_d);
		cudaFree(r_d_d);
		cudaFree(x_d_d);
		cudaFree(data_d_d);
		
		//cudaDeviceSynchronize();
		
		//cudaMemcpy(d, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		for(int i = 0; i < dept; ++i) d[i] = r_d[i];
		std::cout <<  "d " << *d;
		//cudaMemcpy(Ad, Ad_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		for(int i = 0; i < dept; ++i) Ad[i] = Ad_d[i];
		std::cout << "Ad "<<*Ad;
		alpha_cd = delta / mult(d , Ad,  dept);
		//add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
		add_mult(((int) dept/1024) + 1, std::min(1024, dept),x_d,r_d,alpha_cd,dept);
		if(run%50 == 0){
			//cudaMemcpy(r_d, &b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
			for(int i = 0; i < dept; ++i) r_d[i] = b[i];
			//cudaDeviceSynchronize();
			switch(kernel){
				case 0: 
					//kernel_linear<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
					//kernel_linear(grid,block,q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
					//kernel_linear(b, data, datlast, q_d, r_d, x_d, data.back().size(), QA_cost, 1/cost,-1);
					break;
				case 1: 
					//kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
					std::cerr << "Wird noch nicht unterstützt" << std::endl;
					break;
				case 2: 
					//kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
					std::cerr << "Wird noch nicht unterstützt" << std::endl;
					break;
				default: throw std::runtime_error("Can not decide wich kernel!");
			}
			//cudaDeviceSynchronize();	
			//cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
			for(int i = 0; i < dept; ++i) r[i] = r_d[i];
		}else{
			for(int index = 0; index < dept; ++index){
				r[index] -= alpha_cd * Ad[index];
			}
		}
		delta = mult(r , r, dept);
		if(delta < eps * eps * delta0) break;
		beta = -mult(r, Ad, dept) / mult(d, Ad, dept);
		add(mult(beta, d, dept),r, d, dept);
		
		//cudaMemcpy(r_d, d, dept*sizeof(double), cudaMemcpyHostToDevice);
		for(int i = 0; i < dept; ++i) r_d[i] = d[i];
	}
	if(run == imax) std::clog << "Regard reached maximum number of CG-iterations" << std::endl;
	alpha.resize(dept);
	std::vector<double> ret_q(dept);
	// cudaDeviceSynchronize();
	// cudaMemcpy(&alpha[0],x_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < dept; ++i) alpha[i] = x_d[i];
	// cudaMemcpy(&ret_q[0],q_d, dept * sizeof(double), cudaMemcpyDeviceToHost);
	for(int i = 0; i < dept; ++i) ret_q[i] = q_d[i];
	// cudaFree(Ad_d);
	// cudaFree(r_d);
	// cudaFree(datlast);
	// cudaFreeHost(Ad);
	// cudaFree(x_d);
	// cudaFreeHost(r);
	// cudaFreeHost(d);
	delete [] Ad_d, r_d, datlast, Ad, x_d, r, d;
	return ret_q;
}