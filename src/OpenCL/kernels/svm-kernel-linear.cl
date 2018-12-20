//#include "../include/typedef.hpp"

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
void __attribute__((overloadable)) AtomicAdd(__global double *source, double delta) {
    union {
	    double f;
    	ulong i;
    } oldVal;
    union {
    	double f;
		ulong i;
    } newVal;
	// int i = 0;
    do {
    	oldVal.f = *source;
		newVal.f = oldVal.f + delta;
		// ++i;
    } while (atom_cmpxchg ( (volatile __global ulong *)source, oldVal.i, newVal.i) != oldVal.i);
		// if (i > 1) printf("%i\n", i);
}



// void __attribute__((overloadable)) AtomicAdd(__global float *source, float delta) {
//     union {
// 	    float f;
//     	unsigned i;
//     } oldVal;
//     union {
//     	float f;
// 		unsigned i;
//     } newVal;
//     do {
//     	oldVal.f = *source;
// 		newVal.f = oldVal.f + delta;
//     } while (atom_cmpxchg ( (volatile __global unsigned *)source, oldVal.i, newVal.i) != oldVal.i);
// }
__kernel void kernel_linear(__global const real_t *q, __global real_t *ret, __global const real_t *d, __global const real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows, const int add, const int start_block_x, const int start_block_y){  

	int i =  get_group_id(0) * get_local_size(0);
	int j =  get_group_id(1) * get_local_size(1);

	real_t matr = 0.0;

	
	if(i >= j){
		i += 	get_local_id(0);
		j += 	get_local_id(1);
		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
			matr += data_d[vec_index + i] * data_d[vec_index + j];
		}

		const real_t temp = (matr  + QA_cost - q[i] - q[j]) * add;
		if(i > j){
				AtomicAdd(&ret[i], temp * d[j]);
				AtomicAdd(&ret[j], temp * d[i]);
		}else if(i == j){
				AtomicAdd(&ret[j], (temp + cost * add) * d[i]);
		}
		
	}
}