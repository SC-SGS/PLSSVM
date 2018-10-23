#ifndef SVM_KERNEL_H
#define SVM_KERNEL_H

#include "CSVM.hpp"
#include "cuda-kernel.hpp"

/*
void kernel_linear(std::tuple<int,int>, std::tuple<int,int>, double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add);

void kernel_poly(std::tuple<int,int>, std::tuple<int,int>, double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma, const double coef0 ,const double degree);

void kernel_radial(std::tuple<int,int>, std::tuple<int,int>, double *q, double *ret, double *d, double *data_d,const double QA_cost, const double cost,const int Ncols,const int Nrows,const int add, const double gamma);
*/
void kernel_linear(const std::vector<double> &b, std::vector<std::vector<double>> &data, double *datlast, double *q_d, double *Ad, const double *d, const int dim,const double QA_cost, const double cost, const int add);

#endif //SVM_KERNEL_H
