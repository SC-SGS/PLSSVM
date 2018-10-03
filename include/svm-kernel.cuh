#ifndef SVM_KERNEL_H
#define SVM_KERNEL_H

#include "CSVM.hpp"
#include "cuda-kernel.cuh"


__global__  void kernel_linear(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add);

__global__ void kernel_poly(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma, const double coef0 ,const double degree);

__global__  void kernel_radial(double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma);


#endif //SVM_KERNEL_H
