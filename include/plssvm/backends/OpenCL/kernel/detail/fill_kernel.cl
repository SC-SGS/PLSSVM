/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement a fill kernel using OpenCL.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Fill the float data pointer @p data with the @p value.
 * @param[out] data the pointer to fill with values
 * @param[in] value the value used to fill @p data
 * @param[in] pos the start position for filling @p data
 * @param[in] size the number of elements in @p data
 */
__kernel void device_fill_kernel_float(__global float *data, const float value, const ulong pos, const ulong size) {
    const ulong idx = get_global_id(0);
    if (idx < size) {
        data[pos + idx] = value;
    }
}

/**
 * @brief Fill the double data pointer @p data with the @p value.
 * @param[out] data the pointer to fill with values
 * @param[in] value the value used to fill @p data
 * @param[in] pos the start position for filling @p data
 * @param[in] size the number of elements in @p data
 */
__kernel void device_fill_kernel_double(__global double *data, const double value, const ulong pos, const ulong size) {
    const ulong idx = get_global_id(0);
    if (idx < size) {
        data[pos + idx] = value;
    }
}
