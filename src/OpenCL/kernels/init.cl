#include "../include/typedef.hpp"
__kernel void init(__global real_t* vec, real_t value, int size){
	// int id = blockIdx.x * blockDim.x + threadIdx.x;
    int id = get_global_id(0);
	if(id < size) vec[id] = value;
}