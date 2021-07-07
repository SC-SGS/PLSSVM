#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/backends/CUDA/cuda-kernel.hpp>

namespace plssvm {

/*
void kernel_linear(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add);

void kernel_poly(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const real_t degree);

void kernel_radial(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma);
*/
void kernel_linear(const std::vector<real_t> &b,
                   std::vector<std::vector<real_t>> &data,
                   real_t *datlast,
                   const real_t *q_d,
                   std::vector<real_t> &Ad,
                   const real_t *d,
                   const int dim,
                   const real_t QA_cost,
                   const real_t cost,
                   const int add);

}  // namespace plssvm