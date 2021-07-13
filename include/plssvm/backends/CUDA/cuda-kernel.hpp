#pragma once

// TODO: remove if helper functions are all transferred to GPU

namespace plssvm {

template <typename real_type>
void init_(int, int, real_type *vec, real_type value, int size);

template <typename real_type>
void add_mult_(const int, const int, real_type *vec1, const real_type *vec2, const real_type value, const int dim);  // TODO: transfer to multi-GPU

template <typename real_type>
void kernel_q_(int, int, real_type *q, real_type *data_d, real_type *datlast, const int Ncols, const int Nrows);

}  // namespace plssvm