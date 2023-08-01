/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions for explicitly assemblying the kernel matrix using the OpenCL backend.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

/**
 * @brief Create the explicit kernel matrix using the linear kernel function (\f$\vec{u}^T \cdot \vec{v}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 */
__kernel void device_kernel_assembly_linear(__global real_type *ret, __global const real_type *data_d, const ulong num_rows, const ulong num_features, __global const real_type *q, const real_type QA_cost, const real_type cost) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    if (i < num_rows && j < num_rows && i >= j) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; ++dim) {
            temp += data_d[dim * (num_rows + 1) + i] * data_d[dim * (num_rows + 1) + j];
        }
        temp = temp + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

#ifdef PLSSVM_USE_GEMM
        ret[j * num_rows + i] = temp;
        ret[i * num_rows + j] = temp;
#else
        ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
    }
}

/**
 * @brief Create the explicit kernel matrix using the polynomial kernel function (\f$(gamma \cdot \vec{u}^T \cdot \vec{v} + coef0)^{degree}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] degree parameter used in the polynomial kernel function
 * @param[in] gamma parameter used in the polynomial kernel function
 * @param[in] coef0 parameter used in the polynomial kernel function
 */
__kernel void device_kernel_assembly_polynomial(__global real_type *ret, __global const real_type *data_d, const ulong num_rows, const ulong num_features, __global const real_type *q, const real_type QA_cost, const real_type cost, const int degree, const real_type gamma, const real_type coef0) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    if (i < num_rows && j < num_rows && i >= j) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; ++dim) {
            temp += data_d[dim * (num_rows + 1) + i] * data_d[dim * (num_rows + 1) + j];
        }
        temp = pow(gamma * temp + coef0, degree) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

#ifdef PLSSVM_USE_GEMM
        ret[j * num_rows + i] = temp;
        ret[i * num_rows + j] = temp;
#else
        ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
    }
}

/**
 * @brief Create the explicit kernel matrix using the rbf kernel function (\f$e^{(-gamma \cdot |\vec{u} - \vec{v}|^2)}\f$).
 * @param[out] ret the calculated kernel matrix
 * @param[in] data_d the data points to calculate the kernel matrix from
 * @param[in] num_rows the number of data points
 * @param[in] num_features the number of features per data point
 * @param[in] q the vector used in the dimensional reduction
 * @param[in] QA_cost the scalar used in the dimensional reduction
 * @param[in] cost the cost factor the diagonal is scaled with
 * @param[in] gamma parameter used in the rbf kernel function
 */
__kernel void device_kernel_assembly_rbf(__global real_type *ret, __global const real_type *data_d, const ulong num_rows, const ulong num_features, __global const real_type *q, const real_type QA_cost, const real_type cost, const real_type gamma) {
    const ulong i = get_global_id(0);
    const ulong j = get_global_id(1);

    if (i < num_rows && j < num_rows && i >= j) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; ++dim) {
            const real_type d = data_d[dim * (num_rows + 1) + i] - data_d[dim * (num_rows + 1) + j];
            temp += d * d;
        }
        temp = exp(-gamma * temp) + QA_cost - q[i] - q[j];
        if (i == j) {
            temp += cost;
        }

#ifdef PLSSVM_USE_GEMM
        ret[j * num_rows + i] = temp;
        ret[i * num_rows + j] = temp;
#else
        ret[j * num_rows + i - j * (j + 1) / 2] = temp;
#endif
    }
}