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

// based on: https://community.khronos.org/t/roadmap-for-atomic-floating-point-support/7619/5

/**
 * @brief Implementation of an atomic add function for double-precision floating point types.
 * @details Uses the atomic compare-exchange idiom.
 * @param[in,out] source the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global double * source, const double val) {
    union {
        ulong intVal;
        double doubleVal;
    } old, t, t1;
    old.doubleVal = val;
    t1.doubleVal = 0.0; // to ensure correct double bit representation of 0
    do {
        t.intVal = atom_xchg((__global ulong *)source, t1.intVal);
        t.doubleVal += old.doubleVal;
    } while ((old.intVal = atom_xchg((__global ulong *)source, t.intVal)) != t1.intVal);
}

/**
 * @brief Implementation of an atomic add function for single-precision floating point types.
 * @details Uses the atomic compare-exchange idiom.
 * @param[in,out] addr the source value to add @p val to
 * @param[in] val the value to add to @p addr
 */
inline void __attribute__((overloadable)) atomicAdd(__global float * source, const float val) {
    union {
        ulong intVal;
        float floatVal;
    } old, t, t1;
    old.floatVal = val;
    t1.floatVal = 0.0; // to ensure correct double bit representation of 0
    do {
        t.intVal = atom_xchg((__global ulong *)source, t1.intVal);
        t.floatVal += old.floatVal;
    } while ((old.intVal = atom_xchg((__global ulong *)source, t.intVal)) != t1.intVal);
}