#ifndef CUDA_KERNEL_C_H
#define CUDA_KERNEL_C_H

#include "CSVM.hpp"

void init_(int, int, double* vec, double value, int size);
void init_(double* vec, double value, int size);

void add_mult_(int, int, double* vec1, double* vec2, double value, int dim);
void add_mult_(double* vec1, double* vec2, double value, int dim);

void kernel_q_(int, int, double *q, double *data_d, double *datlast,const int Ncols, const int Nrows);
void kernel_q_(double *q, double *data_d, double *datlast,const int Ncols, const int Nrows);



#endif //CUDA_KERNEL_