#pragma once

#if defined(HAS_CPU_BACKEND)
#include <plssvm/OpenMP/CPU_CSVM.hpp>
#endif
#if defined(HAS_CUDA_BACKEND)
#include <plssvm/CUDA/CUDA_CSVM.hpp>
#endif
#if defined(HAS_OPENCL_BACKEND)
#include <plssvm/OpenCL/OCL_CSVM.hpp>
#endif

#include <plssvm/CSVM.hpp>
