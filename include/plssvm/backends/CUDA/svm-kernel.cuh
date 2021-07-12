#pragma once

namespace plssvm {

template <typename real_type>
__global__ void kernel_linear(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end);

template <typename real_type>
__global__ void kernel_poly(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end, const real_type gamma, const real_type coef0, const real_type degree);

template <typename real_type>
__global__ void kernel_radial(const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end, const real_type gamma);

}  // namespace plssvm