#include "cuda-kernel.cuh"

__global__ void init(real_t* vec, real_t value, int size){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size) vec[id] = value;
}

__global__ void add_mult(real_t* vec1, real_t* vec2, real_t value, int dim){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < dim){
		vec1[id] += vec2[id] * value;
	}
}

__global__ void kernel_q_old(real_t *q, real_t *data_d, real_t *datlast,const int Ncols, const int Nrows){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	real_t temp = 0;
	for(int i = 0; i < Ncols ; ++i){
		temp += data_d[i * Nrows + index] * datlast[i];
	}
	q[index] = temp;  
}
__global__ void kernel_q(real_t *q,  real_t *data_d, real_t *datlast, const int Nrows, const int start, const int end){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	real_t temp = 0;
	for(int i = start; i < end ; ++i){
		temp += data_d[i * Nrows + index] * datlast[i];
	}
	q[index] = temp;  //TODO: nachschauen += ?
}