
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
static inline void __attribute__((overloadable)) AtomicAdd(__global const double *source, const double delta) {
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



static inline void __attribute__((overloadable)) AtomicAdd(__global const float *source, const float delta) {
    union {
	    float f;
    	unsigned i;
    } oldVal;
    union {
    	float f;
		unsigned i;
    } newVal;
    do {
    	oldVal.f = *source;
		newVal.f = oldVal.f + delta;
    } while (atom_cmpxchg ( (volatile __global unsigned *)source, oldVal.i, newVal.i) != oldVal.i);
}


__kernel void kernel_linear(__global __read_only const real_t *q, __global __read_write real_t *ret, __global __read_only const real_t *d, __global __read_only const real_t *data_d, __read_only const real_t QA_cost, __read_only const real_t cost,__read_only const int Ncols, __read_only const int Nrows, __read_only const int add, __read_only const int start_block_x, __read_only const int start_block_y){  

	size_t i =  (get_group_id(0) + start_block_x) * (get_local_size(0) * INTERNALBLOCK_SIZE);
	size_t j =  (get_group_id(1) + start_block_y) * (get_local_size(1) * INTERNALBLOCK_SIZE);

	__local real_t data_intern_i [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
	__local real_t data_intern_j [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
	real_t matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
	real_t data_j[INTERNALBLOCK_SIZE];

	
	if(i >= j){
		i += 	get_local_id(0) * INTERNALBLOCK_SIZE;
		const size_t ij = j + get_local_id(0) * INTERNALBLOCK_SIZE;
		j += 	get_local_id(1) * INTERNALBLOCK_SIZE;
		

		//cache data
		for(int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows){
			barrier(CLK_LOCAL_MEM_FENCE);
			#pragma unroll INTERNALBLOCK_SIZE
			for(size_t block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id){
				const size_t idx = block_id % THREADBLOCK_SIZE; //lastbalancieung //TODO: constexpr
				if(get_local_id(1) == idx)data_intern_i[get_local_id(0)][block_id] = data_d[block_id + vec_index + i]; 
				const size_t idx_2 = (block_id + INTERNALBLOCK_SIZE) % THREADBLOCK_SIZE; //lastbalancieung //TODO: constexpr
				if(get_local_id(1) == idx_2)data_intern_j[get_local_id(0)][block_id] = data_d[block_id + vec_index + ij];
			}
			barrier(CLK_LOCAL_MEM_FENCE);

			#pragma unroll INTERNALBLOCK_SIZE
			for(size_t data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index){
				data_j[data_index] = data_intern_j[get_local_id(1)][data_index];
			}
			
			#pragma unroll INTERNALBLOCK_SIZE
			for(size_t l = 0; l < INTERNALBLOCK_SIZE; ++l){
				const real_t data_i =  data_intern_i[get_local_id(0)][l];
				#pragma unroll INTERNALBLOCK_SIZE
				for(size_t k = 0; k < INTERNALBLOCK_SIZE; ++k){
					matr[k][l] += data_i * data_j[k];
				}
			}
		}

		#pragma unroll(INTERNALBLOCK_SIZE) 
		for(size_t k = j; k < INTERNALBLOCK_SIZE + j; ++k){
			const real_t q_j = q[k];
			real_t ret_k = 0;
			#pragma unroll(INTERNALBLOCK_SIZE) 
			for(size_t l = i; l < INTERNALBLOCK_SIZE + i; ++l){
				const real_t temp = (matr[k - j][l - i]  + QA_cost - q[l] - q_j) * add;
				if(l > k){
					AtomicAdd(&ret[l], temp * d[k]);
					ret_k += temp * d[l];
				}else if(l == k){
					ret_k += (temp + cost * add) * d[l];
				}
			}
			AtomicAdd(&ret[k], ret_k);
		}
	}
}


// __kernel void kernel_linear(__global __read_only const real_t *q, __global __read_write real_t *ret, __global __read_only const real_t *d, __global __read_only const real_t *data_d, __read_only const real_t QA_cost, __read_only const real_t cost,__read_only const int Ncols, __read_only const int Nrows, __read_only const int add, __read_only const int start_block_x, __read_only const int start_block_y){  
// 	int i =  get_group_id(0) * (get_local_size(0) * INTERNALBLOCK_SIZE);
// 	int j =  get_group_id(1) * (get_local_size(1) * INTERNALBLOCK_SIZE);

// 	__private real_t matr = 0.0;

	
// 	if(i >= j){
// 		i += 	get_local_id(0) * INTERNALBLOCK_SIZE;
// 		j += 	get_local_id(1) * INTERNALBLOCK_SIZE;
// 		#pragma unroll(INTERNALBLOCK_SIZE) 
// 		for(int k = 0; k < INTERNALBLOCK_SIZE ; ++k){
// 			real_t ret_k = 0;
// 			#pragma unroll(INTERNALBLOCK_SIZE) 
// 			for(int l = i; l < INTERNALBLOCK_SIZE + i; ++l){
// 				matr = 0;
// 				for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 					matr += data_d[vec_index + l] * data_d[vec_index + j];
// 				}

// 				const real_t temp = (matr  + QA_cost - q[l] - q[j]) * add;
// 				if(l > j){
// 					AtomicAdd(&ret[l], temp * d[j]);
// 					ret_k += temp * d[l];
// 				}else if(l == j){
// 					ret_k += (temp + cost * add) * d[l];
// 				}
// 			}
// 			AtomicAdd(&ret[j], ret_k);
// 			j++;
// 		}
// 	}
// }
