/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement a memset kernel using OpenCL.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Memset the float data pointer @p data with the @p value.
 * @param[out] data the pointer to memset with the pattern
 * @param[in] pattern the pattern used to memset @p data
 * @param[in] pos the start position for the memset operation on @p data
 * @param[in] size the number of elements in @p data
 */
__kernel void device_memset_kernel_float(__global float *data, const uchar pattern, const ulong pos, const ulong size) {
    const ulong idx = get_global_id(0);
    if (idx < size) {
        // pack the 1-Byte pattern into a 4-Byte uint
        const uint packed_pattern = (pattern << 24) | (pattern << 16) | (pattern << 8) | pattern;
        // bitwise cast the uint to a float
        data[pos + idx] = as_float(packed_pattern);
    }
}

/**
 * @brief Memset the double data pointer @p data with the @p value.
 * @param[out] data the pointer to memset with the pattern
 * @param[in] pattern the pattern used to memset @p data
 * @param[in] pos the start position for the memset operation on @p data
 * @param[in] size the number of elements in @p data
 */
__kernel void device_memset_kernel_double(__global double *data, const uchar pattern, const ulong pos, const ulong size) {
    const ulong idx = get_global_id(0);
    if (idx < size) {
        // pack the 1-Byte pattern into an 8-Byte ulong
        const ulong packed_pattern = ((ulong) pattern << 56) | ((ulong) pattern << 48) | ((ulong) pattern << 40) | ((ulong) pattern << 32) | ((ulong) pattern << 24) | ((ulong) pattern << 16) | ((ulong) pattern << 8) | ((ulong) pattern);
        // bitwise cast th ulong to a double
        data[pos + idx] = as_double(packed_pattern);
    }
}
