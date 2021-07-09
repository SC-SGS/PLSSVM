#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/typedef.hpp>
//TODO: remove if helperfunctions are all transfered to gpu

namespace plssvm {

void init_(int, int, real_t *vec, real_t value, int size);

template <typename T>
void add_mult_(const int, const int, T *vec1, const T *vec2, const T value, const int dim);  //TODO: transfair to multigpu

void kernel_q_(int, int, real_t *q, real_t *data_d, real_t *datlast, const int Ncols, const int Nrows);

}  // namespace plssvm