#pragma once

#include "device.hpp"
#include "manager.hpp"

namespace opencl {

void run_kernel_1d_timed(device_t &device, cl_kernel kernel, size_t grid_size,
                         size_t block_size);

void run_kernel_2d_timed(device_t &device, cl_kernel kernel, std::vector<size_t> grid_size,
                         std::vector<size_t> block_size);

} // namespace opencl
