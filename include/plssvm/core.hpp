#pragma once

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
#include <plssvm/OpenMP/CPU_CSVM.hpp>
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
#include <plssvm/CUDA/CUDA_CSVM.hpp>
#endif
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
#include <plssvm/OpenCL/OpenCL_CSVM.hpp>
#endif

#include <plssvm/CSVM.hpp>
