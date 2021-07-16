#include "plssvm/backends/CUDA/cuda-kernel.cuh"
#include "plssvm/backends/CUDA/svm-kernel.cuh"

#include "plssvm/typedef.hpp"

namespace plssvm {

template <typename real_type>
__global__ void kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end) {
    unsigned int i = blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
    unsigned int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;

    __shared__ real_type data_intern_i[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    __shared__ real_type data_intern_j[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    real_type matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNALBLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNALBLOCK_SIZE;
        const unsigned int ji = j + threadIdx.x * INTERNALBLOCK_SIZE;
        j += threadIdx.y * INTERNALBLOCK_SIZE;
        // cache data
        for (int vec_index = start * Nrows; vec_index < end * Nrows; vec_index += Nrows) {
            __syncthreads();
            #pragma unroll INTERNALBLOCK_SIZE
            for (std::size_t block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id) {
                const size_t idx = block_id % THREADBLOCK_SIZE;
                if (threadIdx.y == idx) {
                    data_intern_i[threadIdx.x][block_id] = data_d[block_id + vec_index + i];
                }
                const size_t idx_2 = block_id + INTERNALBLOCK_SIZE % THREADBLOCK_SIZE;
                if (threadIdx.y == idx_2) {
                    data_intern_j[threadIdx.x][block_id] = data_d[block_id + vec_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNALBLOCK_SIZE
            for (std::size_t data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }

            #pragma unroll INTERNALBLOCK_SIZE
            for (std::size_t l = 0; l < INTERNALBLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i[threadIdx.x][l];
                #pragma unroll INTERNALBLOCK_SIZE
                for (std::size_t k = 0; k < INTERNALBLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNALBLOCK_SIZE
        for (std::size_t x = 0; x < INTERNALBLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNALBLOCK_SIZE
            for (std::size_t y = 0; y < INTERNALBLOCK_SIZE; ++y) {
                real_type temp;
                if (start == 0) {
                    temp = (matr[x][y] + QA_cost - q[i + y] - q[j + x]) * add;
                } else {
                    temp = matr[x][y] * add;
                }
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomicAdd(&ret[i + y], temp * d[j + x]);
                    ret_jx += temp * d[i + y];
                    // atomicAdd(&ret[j + x], temp * d[i + y]);
                } else if (i + x == j + y) {
                    // diagonal
                    if (start == 0) {
                        ret_jx += (temp + cost * add) * d[i + y];
                        // atomicAdd(&ret[j + x], (temp + cost * add) * d[i + y]);
                    } else {
                        ret_jx += temp * d[i + y];
                        // atomicAdd(&ret[j + x], temp * d[i + y]);
                    }
                }
            }
            atomicAdd(&ret[j + x], ret_jx);
        }
    }
}

template __global__ void kernel_linear(const float *, float *, const float *, const float *, const float, const float, const int, const int, const int, const int, const int);
template __global__ void kernel_linear(const double *, double *, const double *, const double *, const double, const double, const int, const int, const int, const int, const int);

template <typename real_type>  // TODO: remove start / end ?
__global__ void kernel_poly(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end, const real_type gamma, const real_type coef0, const real_type degree) {
    unsigned int i = blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
    unsigned int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;

    __shared__ real_type data_intern_i[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    __shared__ real_type data_intern_j[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    real_type matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNALBLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNALBLOCK_SIZE;
        const unsigned int ji = j + threadIdx.x * INTERNALBLOCK_SIZE;
        j += threadIdx.y * INTERNALBLOCK_SIZE;
        for (int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows) {
            {
                #pragma unroll INTERNALBLOCK_SIZE
                for (int block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id) {
                    const int data_index = vec_index + block_id;
                    if (threadIdx.y == block_id)
                        data_intern_i[threadIdx.x][block_id] = data_d[data_index + i];
                    if (threadIdx.y == block_id * 2)
                        data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll INTERNALBLOCK_SIZE
            for (int data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }
            __syncthreads();
            #pragma unroll INTERNALBLOCK_SIZE
            for (int x = 0; x < INTERNALBLOCK_SIZE; ++x) {
                const real_type data_i = data_intern_i[threadIdx.x][x];
                #pragma unroll INTERNALBLOCK_SIZE
                for (int y = 0; y < INTERNALBLOCK_SIZE; ++y) {
                    matr[x][y] += data_i * data_j[y];
                }
            }
        }
        #pragma unroll INTERNALBLOCK_SIZE
        for (int x = 0; x < INTERNALBLOCK_SIZE; ++x) {
            #pragma unroll INTERNALBLOCK_SIZE
            for (int y = 0; y < INTERNALBLOCK_SIZE; ++y) {
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

template __global__ void kernel_poly(const float *, float *, const float *, const float *, const float, const float, const int, const int, const int, const int, const int, const float, const float, const float);
template __global__ void kernel_poly(const double *, double *, const double *, const double *, const double, const double, const int, const int, const int, const int, const int, const double, const double, const double);

template <typename real_type>
__global__ void kernel_radial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end, const real_type gamma) {
    int i = blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
    int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;

    __shared__ real_type data_intern_i[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    __shared__ real_type data_intern_j[THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
    real_type matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
    real_type data_j[INTERNALBLOCK_SIZE];

    if (i >= j) {
        i += threadIdx.x * INTERNALBLOCK_SIZE;
        const int ji = j + threadIdx.x * INTERNALBLOCK_SIZE;
        j += threadIdx.y * INTERNALBLOCK_SIZE;
        for (int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows) {
            {
                #pragma unroll(INTERNALBLOCK_SIZE)
                for (int block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id) {
                    const int data_index = vec_index + block_id;
                    if (threadIdx.y == block_id)
                        data_intern_i[threadIdx.x][block_id] = data_d[data_index + i];
                    if (threadIdx.y == block_id * 2)
                        data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
                }
            }
            __syncthreads();

            #pragma unroll(INTERNALBLOCK_SIZE)
            for (int data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j[threadIdx.y][data_index];
            }
            __syncthreads();
            #pragma unroll(INTERNALBLOCK_SIZE)
            for (int x = 0; x < INTERNALBLOCK_SIZE; ++x) {
                const real_type data_i = data_intern_i[threadIdx.x][x];
                #pragma unroll(INTERNALBLOCK_SIZE)
                for (int y = 0; y < INTERNALBLOCK_SIZE; ++y) {
                    matr[x][y] += (data_i - data_j[y]) * (data_i - data_j[y]);
                }
            }
        }

        #pragma unroll(INTERNALBLOCK_SIZE)
        for (int x = 0; x < INTERNALBLOCK_SIZE; ++x) {
            #pragma unroll(INTERNALBLOCK_SIZE)
            for (int y = 0; y < INTERNALBLOCK_SIZE; ++y) {
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
template __global__ void kernel_radial(const float *, float *, const float *, const float *, const float, const float, const int, const int, const int, const int, const int, const float);
template __global__ void kernel_radial(const double *, double *, const double *, const double *, const double, const double, const int, const int, const int, const int, const int, const double);

}  // namespace plssvm