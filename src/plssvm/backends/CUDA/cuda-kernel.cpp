#include "plssvm/backends/CUDA/cuda-kernel.hpp"

namespace plssvm {

template <typename real_type>
void init_(int block, int blockDim, real_type *vec, real_type value, int size) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int id = blockIdx * blockDim + threadIdx;
            if (id < size)
                vec[id] = value;
        }
    }
}
template void init_(int, int, float *, float, int);
template void init_(int, int, double *, double, int);

template <typename real_type>
void add_mult_(const int block, const int blockDim, real_type *vec1, const real_type *vec2, const real_type value, const int dim) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int id = blockIdx * blockDim + threadIdx;
            if (id < dim) {
                vec1[id] += vec2[id] * value;
            }
        }
    }
}
template void add_mult_(const int, const int, float *, const float *, const float, const int);
template void add_mult_(const int, const int, double *, const double *, const double, const int);

template <typename real_type>
void kernel_q_(int block, int blockDim, real_type *q, real_type *data_d, real_type *datlast, const int Ncols, const int Nrows) {
    for (int blockIdx = 0; blockIdx < block; ++blockIdx) {
        for (int threadIdx = 0; threadIdx < blockDim; ++threadIdx) {
            int index = blockIdx * blockDim + threadIdx;
            real_type temp = 0;
            for (int i = 0; i < Ncols; ++i) {
                temp += data_d[i * Nrows + index] * datlast[i];
            }
            q[index] = temp;
        }
    }
}
template void kernel_q_(int, int, float *, float *, float *, const int, const int);
template void kernel_q_(int, int, double *, double *, double *, const int, const int);

}  // namespace plssvm