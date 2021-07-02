#include <plssvm/CUDA/cuda-kernel.hpp>

namespace plssvm {

void init_(int block, int blockDim, real_t *vec, real_t value, int size) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int id = blockIdx * blockDim + threadIdx;
            if (id < size)
                vec[id] = value;
        }
    }
}

void add_mult_(const int block, const int blockDim, real_t *vec1, const real_t *vec2, const real_t value, const int dim) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int id = blockIdx * blockDim + threadIdx;
            if (id < dim) {
                vec1[id] += vec2[id] * value;
            }
        }
    }
}

void kernel_q_(int block, int blockDim, real_t *q, real_t *data_d, real_t *datlast, const int Ncols, const int Nrows) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int index = blockIdx * blockDim + threadIdx;
            real_t temp = 0;
            for (int i = 0; i < Ncols; ++i) {
                temp += data_d[i * Nrows + index] * datlast[i];
            }
            q[index] = temp;
        }
    }
}

}  // namespace plssvm