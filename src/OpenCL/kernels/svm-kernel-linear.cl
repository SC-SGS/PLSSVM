__kernel void kernel_linear(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add){
	int i =  blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
	int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;

	__shared__ double data_intern_i [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
	__shared__ double data_intern_j [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
	double matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
	double data_j[BLOCKING_SIZE_THREAD];

	
	if(i >= j){
		i += threadIdx.x * BLOCKING_SIZE_THREAD;
		const int ji = j +  threadIdx.x * BLOCKING_SIZE_THREAD;
		j += threadIdx.y * BLOCKING_SIZE_THREAD;
		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
			{
				//#pragma unroll(BLOCKING_SIZE_THREAD)
				for(int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id){
					const int data_index = vec_index + block_id;
					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];  
					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
				}

			}
			__syncthreads();

			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index){
				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
			}
			__syncthreads();
			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
				const double data_i = data_intern_i[threadIdx.x][x];				
			//	#pragma unroll(BLOCKING_SIZE_THREAD)
				for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
					matr[x][y] += data_i * data_j[y];
				}
			}
		}
		//#pragma unroll(BLOCKING_SIZE_THREAD)
		for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
				const double temp = (matr[x][y]  + QA_cost - q[i + x] - q[j + y]) * add;
				if(i + x > j + y){
					atomicAdd(&ret[i + x], temp * d[j + y]);
					atomicAdd(&ret[j + y], temp * d[i + x]);
				}else if(i + x == j + y){
					atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
				}
			}
		}
	}
}