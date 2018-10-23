#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

#include "CSVM.hpp"

void init(int, int, double* vec, double value, int size);
void init(double* vec, double value, int size);

void add_mult(int, int, double* vec1, double* vec2, double value, int dim);
void add_mult(double* vec1, double* vec2, double value, int dim);

void kernel_q(int, int, double *q, double *data_d, double *datlast,const int Ncols, const int Nrows);
void kernel_q(double *q, double *data_d, double *datlast,const int Ncols, const int Nrows);



#endif //CUDA_KERNEL_