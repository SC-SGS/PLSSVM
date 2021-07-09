#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::not_implemented_exception

#include <vector>  // std::vector

namespace plssvm {

template <typename T>
void device_kernel_linear(const std::vector<std::vector<T>> &data, std::vector<T> &ret, const std::vector<T> &d, T QA_cost, T cost, int sign);

template <typename T>
void device_kernel_poly(const std::vector<std::vector<T>> &, std::vector<T> &, const std::vector<T> &, T, T, int) {
    throw not_implemented_exception{ "The polynomial kernel is currently not implemented for the OpenMP backend!" };
}

template <typename T>
void device_kernel_radial(const std::vector<std::vector<T>> &, std::vector<T> &, const std::vector<T> &, T, T, int) {
    throw not_implemented_exception{ "The radial basis function kernel is currently not implemented for the OpenMP backend!" };
}

//template <typename T>
//void kernel_linear(const std::vector<T> &b, const std::vector<std::vector<T>> &data, const T *datlast, const T *q_d, std::vector<T> &Ad, const T *d, std::size_t dim, T QA_cost, T cost, int sgn);

/*
void kernel_linear(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add);
void kernel_poly(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const real_t degree);
void kernel_radial(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma);
*/

}  // namespace plssvm