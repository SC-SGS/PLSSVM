#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/CUDA/cuda-kernel.cuh>

namespace plssvm {

__global__ void
kernel_linear(const real_t *q, real_t *ret, const real_t *d, const real_t *data_d, const real_t QA_cost, const real_t cost, const int Ncols, const int Nrows, const int add, const int start, const int end);
// __global__ void kernel_linear(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add);

__global__ void
kernel_poly(real_t *q, real_t *ret, real_t *d, real_t *data_d, const real_t QA_cost, const real_t cost, const int Ncols, const int Nrows, const int add, const real_t gamma, const real_t coef0, const real_t degree);

__global__ void
kernel_radial(real_t *q, real_t *ret, real_t *d, real_t *data_d, const real_t QA_cost, const real_t cost, const int Ncols, const int Nrows, const int add, const real_t gamma);

}  // namespace plssvm