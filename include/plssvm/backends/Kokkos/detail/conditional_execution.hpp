/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Conditionally defined macros for the different available Kokkos ExecutionSpaces.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_CONDITIONAL_EXECUTION_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_CONDITIONAL_EXECUTION_HPP_
#pragma once

#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space

#include "Kokkos_Core.hpp"  // Kokkos macros

#include "fmt/core.h"  // fmt::format

#include <functional>  // std::invoke

namespace plssvm::kokkos::detail {

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA` macro if `KOKKOS_ENABLE_CUDA` is defined, i.e., the Kokkos CUDA ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_CUDA` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_CUDA)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::cuda) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP` macro if `KOKKOS_ENABLE_HIP` is defined, i.e., the Kokkos HIP ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_HIP` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_HIP)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::hip) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL` macro if `KOKKOS_ENABLE_SYCL` is defined, i.e., the Kokkos SYCL ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_SYCL` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_SYCL)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::sycl) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX` macro if `KOKKOS_ENABLE_HPX` is defined, i.e., the Kokkos HPX ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_HPX` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_HPX)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::hpx) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP` macro if `KOKKOS_ENABLE_OPENMP` is defined, i.e., the Kokkos OpenMP ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_OPENMP` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_OPENMP)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::openmp) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET` macro if `KOKKOS_ENABLE_OPENMPTARGET` is defined, i.e., the Kokkos OpenMP target offloading ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_OPENMPTARGET` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::openmp_target) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC` macro if `KOKKOS_ENABLE_OPENACC` is defined, i.e., the Kokkos OpenACC ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_OPENACC` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_OPENACC)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::openacc) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS` macro if `KOKKOS_ENABLE_THREADS` is defined, i.e., the Kokkos std::thread ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_THREADS` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 */
#if defined(KOKKOS_ENABLE_THREADS)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::threads) }
#endif

/**
 * @def PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL
 * @brief Defines the `PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL` macro if `KOKKOS_ENABLE_SERIAL` is defined, i.e., the Kokkos serial ExecutionSpace is available.
 * @details If `KOKKOS_ENABLE_SERIAL` is enabled, invokes the provided function (normally a lambda function), otherwise throws an exception.
 * @note This ExecutionSpace *should* always be available!
 */
#if defined(KOKKOS_ENABLE_SERIAL)
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL(func) return std::invoke(func)
#else
    #define PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL(func) \
        throw backend_exception { fmt::format("The Kokkos ExecutionSpace {} is not available!", execution_space::serial) }
#endif

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_CONDITIONAL_EXECUTION_HPP_
