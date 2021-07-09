#include "plssvm/backends/CUDA/cuda-kernel.cuh"
#include "plssvm/backends/CUDA/svm-kernel.cuh"

namespace plssvm {

template <typename real_type>
__global__ void kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end) {
    int i = blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
    int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;

    __shared__ real_type data_intern_i[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    __shared__ real_type data_intern_j[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    real_type matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
    real_type data_j[INTERNALBLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNALBLOCK_SIZE;
        const int ji = j + threadIdx.x * BLOCKING_SIZE_THREAD;
        j += threadIdx.y * INTERNALBLOCK_SIZE;
        //cache data
        for (int vec_index = start * Nrows; vec_index < end * Nrows; vec_index += Nrows) {
            __syncthreads();
            #pragma unroll(INTERNALBLOCK_SIZE)
            for (size_t block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id) {
                const size_t idx = block_id % THREADBLOCK_SIZE;
                if (threadIdx.y == idx)
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                const size_t idx_2 = block_id + INTERNALBLOCK_SIZE % THREADBLOCK_SIZE;  // TODO: constexpr
                if (threadIdx.y == idx_2)
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
            }
            __syncthreads();

            #pragma unroll INTERNALBLOCK_SIZE
            for (size_t data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNALBLOCK_SIZE
            for (size_t l = 0; l < INTERNALBLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
                #pragma unroll INTERNALBLOCK_SIZE
                for (size_t k = 0; k < INTERNALBLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll(INTERNALBLOCK_SIZE)
        for (int x = 0; x < INTERNALBLOCK_SIZE; ++x) {
            #pragma unroll(INTERNALBLOCK_SIZE)
            for (int y = 0; y < INTERNALBLOCK_SIZE; ++y) {
                real_type temp;
                if (start == 0) {  // TODO: auslagern
                    temp = (matr[x][y] + QA_cost - q[i + y] - q[j + x]) * add;
                } else {
                    temp = matr[x][y] * add;
                }
                if (i + x > j + y) {
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    atomicAdd(&ret[j + x], temp * d[i + y]);
                } else if (i + x == j + y) {
                    if (start == 0) {  // TODO: auslagern
                        atomicAdd(&ret[j + x], (temp + cost * add) * d[i + y]);
                    } else {
                        atomicAdd(&ret[j + x], temp * d[i + y]);
                    }
                }
            }
        }

        // #pragma unroll(INTERNALBLOCK_SIZE)
        // for(size_t k = j; k < INTERNALBLOCK_SIZE + j; ++k){
        // 	const real_type q_j = q[k];
        // 	// real_type ret_k = 0.0;
        // 	#pragma unroll(INTERNALBLOCK_SIZE)
        // 	for(size_t l = i; l < INTERNALBLOCK_SIZE + i; ++l){
        // 		// real_type temp;
        // 		//if(start == 0){
        // 		//temp = (matr[k - j][l - i]  + QA_cost - q[l] - q_j) * add;
        // 		//}else{
        // 		const real_type temp = matr[k - j][l - i] * add;
        // 		//}
        // 		if(l > k){
        // 			atomicAdd(&ret[l], temp * d[k]);
        // 			atomicAdd(&ret[k], temp * d[l]);
        // 		}else if(l == k){
        // 			// if(start == 0){
        // 				atomicAdd(&ret[k], (temp + cost * add) * d[l]);
        // 			// }else{
        // 				// ret_k += temp * d[l];
        // 			// }
        // 		}
        // 	}
        // 	// atomicAdd(&ret[k], ret_k);
        // }
    }
}

template __global__ void kernel_linear(const float *, float *, const float *, const float *, const float, const float, const int, const int, const int, const int, const int);
template __global__ void kernel_linear(const double *, double *, const double *, const double *, const double, const double, const int, const int, const int, const int, const int);

template <typename real_type>  // TODO: start / end ?
__global__ void kernel_poly(real_type *q, real_type *ret, real_type *d, real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const real_type gamma, const real_type coef0, const real_type degree) {
    int i = blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
    int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;

    __shared__ real_type data_intern_i[CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
    __shared__ real_type data_intern_j[CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
    real_type matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
    real_type data_j[BLOCKING_SIZE_THREAD];

    if (i >= j) {
        i += threadIdx.x * BLOCKING_SIZE_THREAD;
        const int ji = j + threadIdx.x * BLOCKING_SIZE_THREAD;
        j += threadIdx.y * BLOCKING_SIZE_THREAD;
        for (int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows) {
            {
                #pragma unroll(BLOCKING_SIZE_THREAD)
                for (int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id) {
                    const int data_index = vec_index + block_id;
                    if (threadIdx.y == block_id)
                        data_intern_i[threadIdx.x][block_id] = data_d[data_index + i];
                    if (threadIdx.y == block_id * 2)
                        data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }
            __syncthreads();
            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int x = 0; x < BLOCKING_SIZE_THREAD; ++x) {
                const real_type data_i = data_intern_i[threadIdx.x][x];
                #pragma unroll(BLOCKING_SIZE_THREAD)
                for (int y = 0; y < BLOCKING_SIZE_THREAD; ++y) {
                    matr[x][y] += data_i * data_j[y];
                }
            }
        }
        #pragma unroll(BLOCKING_SIZE_THREAD)
        for (int x = 0; x < BLOCKING_SIZE_THREAD; ++x) {
            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int y = 0; y < BLOCKING_SIZE_THREAD; ++y) {
                const real_type temp = (pow(gamma * matr[x][y] + coef0, degree) + QA_cost - q[i + x] - q[j + y]) * add;
                if (i + x > j + y) {
                    atomicAdd(&ret[i + x], temp * d[j + y]);
                    atomicAdd(&ret[j + y], temp * d[i + x]);
                } else if (i + x == j + y) {
                    atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
                }
            }
        }
    }
}

template __global__ void kernel_poly(float *, float *, float *, float *, const float, const float, const int, const int, const int, const float, const float, const float);
template __global__ void kernel_poly(double *, double *, double *, double *, const double, const double, const int, const int, const int, const double, const double, const double);

template <typename real_type>
__global__ void kernel_radial(real_type *q, real_type *ret, real_type *d, real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const real_type gamma) {
    int i = blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
    int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;

    __shared__ real_type data_intern_i[CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
    __shared__ real_type data_intern_j[CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
    real_type matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
    real_type data_j[BLOCKING_SIZE_THREAD];

    if (i >= j) {
        i += threadIdx.x * BLOCKING_SIZE_THREAD;
        const int ji = j + threadIdx.x * BLOCKING_SIZE_THREAD;
        j += threadIdx.y * BLOCKING_SIZE_THREAD;
        for (int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows) {
            {
                #pragma unroll(BLOCKING_SIZE_THREAD)
                for (int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id) {
                    const int data_index = vec_index + block_id;
                    if (threadIdx.y == block_id)
                        data_intern_i[threadIdx.x][block_id] = data_d[data_index + i];
                    if (threadIdx.y == block_id * 2)
                        data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }
            __syncthreads();
            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int x = 0; x < BLOCKING_SIZE_THREAD; ++x) {
                const real_type data_i = data_intern_i[threadIdx.x][x];
                #pragma unroll(BLOCKING_SIZE_THREAD)
                for (int y = 0; y < BLOCKING_SIZE_THREAD; ++y) {
                    matr[x][y] += (data_i - data_j[y]) * (data_i - data_j[y]);
                }
            }
        }

        #pragma unroll(BLOCKING_SIZE_THREAD)
        for (int x = 0; x < BLOCKING_SIZE_THREAD; ++x) {
            #pragma unroll(BLOCKING_SIZE_THREAD)
            for (int y = 0; y < BLOCKING_SIZE_THREAD; ++y) {
                const real_type temp = (exp(-gamma * matr[x][y]) + QA_cost - q[i + x] - q[j + y]) * add;
                if (i + x > j + y) {
                    atomicAdd(&ret[i + x], temp * d[j + y]);
                    atomicAdd(&ret[j + y], temp * d[i + x]);
                } else if (i + x == j + y) {
                    atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
                }
            }
        }
    }
}
template __global__ void kernel_radial(float *, float *, float *, float *, const float, const float, const int, const int, const int, const float);
template __global__ void kernel_radial(double *, double *, double *, double *, const double, const double, const int, const int, const int, const double);

}  // namespace plssvm