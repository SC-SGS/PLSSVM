#include "cuda-kernel.cuh"

__global__ void init(real_t* vec, real_t value, int size){
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < size) vec[id] = value;
}

__global__ void add_mult(real_t* vec1, const real_t* vec2, const real_t value, const int dim){
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < dim){
		vec1[id] += vec2[id] * value;
	}
}

__global__ void add_inplace(real_t* vec1, const real_t* vec2, const int dim){
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < dim){
		vec1[id] += vec2[id];
	}
}


__global__ void dot( real_t *a, real_t *b, real_t *c, const int dim ) {
	__shared__ real_t temp[THREADS_PER_BLOCK];
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dim){
		temp[threadIdx.x] = a[index] * b[index];
		
	}else{
		temp[threadIdx.x] = 0.0;
	}
	__syncthreads();
	if( 0 == threadIdx.x ) {
		real_t sum = 0;
		for( int i = 0; i < THREADS_PER_BLOCK; i++ ){
			sum += temp[i];
		}
		atomicAdd( c , sum );
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
__global__ void kernel_q(real_t *q, const real_t *data_d, const real_t *datlast, const int Nrows, const int start, const int end){
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	real_t temp = 0;
	for(int i = start; i < end ; ++i){
		temp += data_d[i * Nrows + index] * datlast[i];
	}
	q[index] = temp;  //TODO: nachschauen += ?
}