#include "cuda-kernel.hpp"



void init_(int block, int blockDim, double* vec, double value, int size){
    for(int blockIdx = 0; blockIdx < block; ++ blockIdx){
        for(int threadIdx = 0; threadIdx < blockDim; ++ threadIdx){
            int id = blockIdx * blockDim + threadIdx;
	        if(id < size) vec[id] = value;
        }
    }
}

void add_mult_(int block, int blockDim, double* vec1, double* vec2, double value, int dim){
    for(int blockIdx = 0; blockIdx < block; ++ blockIdx){
        for(int threadIdx = 0; threadIdx < blockDim; ++ threadIdx){
            int id = blockIdx * blockDim + threadIdx;
	        if(id < dim){
	    	    vec1[id] += vec2[id] * value;
            }
        }
	}
}

void kernel_q_(int block, int blockDim, double *q, double *data_d, double *datlast,const int Ncols, const int Nrows){
	for(int blockIdx = 0; blockIdx < block; ++ blockIdx){
        for(int threadIdx = 0; threadIdx < blockDim; ++ threadIdx){
            int index = blockIdx * blockDim + threadIdx;
            double temp = 0;
            for(int i = 0; i < Ncols ; ++i){
                temp += data_d[i * Nrows + index] * datlast[i];
            }
            q[index] = temp;  
        }
    }
}