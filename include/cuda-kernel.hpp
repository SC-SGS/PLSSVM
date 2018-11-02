#ifndef CUDA_KERNEL_C_H
#define CUDA_KERNEL_C_H

#include "CSVM.hpp"

void init_(int, int, real_t* vec, real_t value, int size);
void init_(real_t* vec, real_t value, int size);

void add_mult_(int, int, real_t* vec1, real_t* vec2, real_t value, int dim);
void add_mult_(real_t* vec1, real_t* vec2, real_t value, int dim);

void kernel_q_(int, int, real_t *q, real_t *data_d, real_t *datlast,const int Ncols, const int Nrows);
void kernel_q_(real_t *q, real_t *data_d, real_t *datlast,const int Ncols, const int Nrows);



#endif //CUDA_KERNEL_