/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines atomic functions for floating point types.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

/**
 * @brief Implementation of an atomic add function for double-precision floating point types.
 * @details Uses the atomic compare-exchange idiom.
 * @param[in,out] addr the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global const double *addr, const double val) {
    union {
        ulong u64;
        double f64;
    } next, expected, current;
    current.f64 = *addr;
    do {
        expected.f64 = current.f64;
        next.f64 = expected.f64 + val;
        current.u64 = atom_cmpxchg((volatile __global ulong *) addr,
                                   expected.u64,
                                   next.u64);
    } while (current.u64 != expected.u64);
}

/**
 * @brief Implementation of an atomic add function for single-precision floating point types.
 * @details Uses the atomic compare-exchange idiom.
 * @param[in,out] addr the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global const float *addr, const float val) {
    union {
        unsigned int u32;
        float f32;
    } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *) addr,
                                     expected.u32,
                                     next.u32);
    } while (current.u32 != expected.u32);
}