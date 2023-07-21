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
    const ulong j_cached_idx = get_group_id(1) * get_local_size(1) + get_local_id(0);

    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE) {
                data_cache_i[get_local_id(1)][get_local_id(0)] = 0.0;
                data_cache_j[get_local_id(1)][get_local_id(0)] = 0.0;
            }

            // load data into shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE && dim + get_local_id(1) < num_features) {
                if (i < num_rows) {
                    data_cache_i[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + j_cached_idx];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (ulong block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += data_cache_i[block_dim][get_local_id(0)] * data_cache_j[block_dim][get_local_id(1)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = temp + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
        }
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
    const ulong j_cached_idx = get_group_id(1) * get_local_size(1) + get_local_id(0);

    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE) {
                data_cache_i[get_local_id(1)][get_local_id(0)] = 0.0;
                data_cache_j[get_local_id(1)][get_local_id(0)] = 0.0;
            }

            // load data into shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE && dim + get_local_id(1) < num_features) {
                if (i < num_rows) {
                    data_cache_i[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + j_cached_idx];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (ulong block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                temp += data_cache_i[block_dim][get_local_id(0)] * data_cache_j[block_dim][get_local_id(1)];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = pow(gamma * temp + coef0, degree) + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
        }
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
    const ulong j_cached_idx = get_group_id(1) * get_local_size(1) + get_local_id(0);

    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp = 0.0;
        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // zero out shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE) {
                data_cache_i[get_local_id(1)][get_local_id(0)] = 0.0;
                data_cache_j[get_local_id(1)][get_local_id(0)] = 0.0;
            }

            // load data into shared memory
            if (get_local_id(1) < FEATURE_BLOCK_SIZE && dim + get_local_id(1) < num_features) {
                if (i < num_rows) {
                    data_cache_i[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + i];
                }
                if (j_cached_idx < num_rows) {
                    data_cache_j[get_local_id(1)][get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1) + j_cached_idx];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (ulong block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                const real_type d = data_cache_i[block_dim][get_local_id(0)] - data_cache_j[block_dim][get_local_id(1)];
                temp += d * d;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (i < num_rows && j < num_rows && i >= j) {
            temp = exp(-gamma * temp) + QA_cost - q[i] - q[j];
            if (i == j) {
                temp += cost;
            }

            ret[j * num_rows + i - j * (j + 1) / 2] = temp;
        }
    }
}