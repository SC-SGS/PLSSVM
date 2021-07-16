#pragma once

namespace plssvm {

template <typename real_type>
__global__ void kernel_q_linear(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int start, const int end);
template <typename real_type>
__global__ void kernel_q_poly(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type degree, const real_type gamma, const real_type coef0);
template <typename real_type>
__global__ void kernel_q_radial(real_type *q, const real_type *data_d, const real_type *datlast, const int Nrows, const int Ncols, const real_type gamma);

}  // namespace plssvm