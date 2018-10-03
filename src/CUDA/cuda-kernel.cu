#include "cuda-kernel.cuh"

__global__ void init(double* vec, double value, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size) vec[id] = value;
}

__global__ void add_mult(double* vec1, double* vec2, double value, int dim){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < dim){
		vec1[id] += vec2[id] * value;
	}
}

__global__ void kernel_q(double *q, double *data_d, double *datlast,const int Ncols, const int Nrows){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
		double temp = 0;
		for(int i = 0; i < Ncols ; ++i){
			 temp += data_d[i * Nrows + index] * datlast[i];
		}
		q[index] = temp;  
}