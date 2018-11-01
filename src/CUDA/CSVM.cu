#include "CSVM.hpp"
#include "cuda-kernel.cuh"
#include "svm-kernel.cuh"

int CUDADEVICE = 0;

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){}

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
	cudaMalloc((void **) &datlast, (Nfeatures_data + CUDABLOCK_SIZE - 1) * sizeof(double));
	cudaMemset(datlast, 0, Nfeatures_data + CUDABLOCK_SIZE - 1 * sizeof(double));
	cudaMemcpy(datlast,&data[Ndatas_data - 1][0], Nfeatures_data * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &data_d, Nfeatures_data * (Ndatas_data + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) -1)* sizeof(double));

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
	dim3 block(CUDABLOCK_SIZE, CUDABLOCK_SIZE);
	double *x_d, *r, *d, *r_d, *q_d;

	cudaMalloc((void **) &x_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)*sizeof(double));
	init<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,1,dept);
	init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1>>>(x_d + dept, 0 , (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);

	//r = b - (A * x)
	///r = b;
	cudaMallocHost((void **) &r, dept *sizeof(double));
	cudaMalloc((void **) &r_d, (dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) *sizeof(double));
	init<<< 1,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD)>>>(r_d + dept, 0 ,(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1);
	cudaMallocHost((void **) &d, dept *sizeof(double));

	cudaMemcpy(r_d,&b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaMalloc((void **) &q_d, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
	cudaMemset(q_d, 0, (dept +  (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  * sizeof(double));
	kernel_q<<<((int) dept/CUDABLOCK_SIZE) + 1, std::min((int)CUDABLOCK_SIZE, dept)>>>(q_d, data_d, datlast, Nfeatures_data , dept + (CUDABLOCK_SIZE * BLOCKING_SIZE_THREAD) );
	switch(kernel){
		case 0: 
			kernel_linear<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
			break;
		case 1: 
			kernel_poly<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
			break;	
		case 2: 
			kernel_radial<<<grid,block>>>(q_d, r_d, x_d,data_d, QA_cost, 1/cost, Nfeatures_data , dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma);
			break;
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
 
	cudaDeviceSynchronize();
	cudaMemcpy(r, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
	double delta = mult(r, r, dept);	
	const double delta0 = delta;
	double alpha_cd, beta;
	double* Ad, *Ad_d;
	cudaMallocHost((void **) &Ad, dept *sizeof(double));
	cudaMalloc((void **) &Ad_d, (dept +(CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1)  *sizeof(double));

	int run;
	for(run = 0; run < imax ; ++run){
		if(info)std::cout << "Start Iteration: " << run << std::endl;
		//Ad = A * d
		cudaMemset(Ad_d, 0, dept * sizeof(double));
		cudaMemset(r_d + dept, 0, ((CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) - 1) * sizeof(double));
	
		switch(kernel){
			case 0: 
				kernel_linear<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1);
				break;
			case 1: 
				kernel_poly<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , 1, gamma, coef0, degree);
				break;
			case 2: 
				kernel_radial<<<grid,block>>>(q_d, Ad_d, r_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), 1, gamma);
				break;
			default: throw std::runtime_error("Can not decide wich kernel!");
		}
		
		cudaDeviceSynchronize();
	
		cudaMemcpy(d, r_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ad, Ad_d, dept*sizeof(double), cudaMemcpyDeviceToHost);
		alpha_cd = delta / mult(d , Ad,  dept);
		add_mult<<< ((int) dept/1024) + 1, std::min(1024, dept)>>>(x_d,r_d,alpha_cd,dept);
		if(run%50 == 0){
			cudaMemcpy(r_d, &b[0], dept * sizeof(double), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();
			switch(kernel){
				case 0: 
					kernel_linear<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1);
					break;
				case 1: 
					kernel_poly<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD), -1, gamma, coef0, degree);
					break;
				case 2: 
					kernel_radial<<<grid,block>>>(q_d, r_d, x_d, data_d, QA_cost, 1/cost, Nfeatures_data, dept + (CUDABLOCK_SIZE*BLOCKING_SIZE_THREAD) , -1, gamma);
					break;
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
		add(mult(beta, d, dept),r, d, dept);
		cudaMemcpy(r_d, d, dept*sizeof(double), cudaMemcpyHostToDevice);
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