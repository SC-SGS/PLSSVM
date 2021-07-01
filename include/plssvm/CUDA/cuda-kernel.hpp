#pragma once

#include <plssvm/CSVM.hpp>

namespace plssvm {

void init_(int, int, real_t *vec, real_t value, int size);
void init_(real_t *vec, real_t value, int size);

void add_mult_(const int, const int, real_t *vec1, const real_t *vec2, const real_t value, const int dim);
void add_mult_(real_t *vec1, const real_t *vec2, const real_t value, const int dim);

void kernel_q_(int, int, real_t *q, real_t *data_d, real_t *datlast, const int Ncols, const int Nrows);
void kernel_q_(real_t *q, real_t *data_d, real_t *datlast, const int Ncols, const int Nrows);

} // namespace plssvm