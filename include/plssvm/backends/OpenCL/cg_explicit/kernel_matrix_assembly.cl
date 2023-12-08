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
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);


    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const ulong global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_i[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_i[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_j[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_j];
                data_cache_j[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] +=  data_cache_i[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = i + internal_i;
                const ulong global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = temp_ij + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#ifdef PLSSVM_USE_GEMM
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
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
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);


    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const ulong global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_i[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_i[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_j[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_j];
                data_cache_j[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        temp[internal_i][internal_j] +=  data_cache_i[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] * data_cache_j[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = i + internal_i;
                const ulong global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = pow(gamma * temp_ij + coef0, degree) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#ifdef PLSSVM_USE_GEMM
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
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
    const ulong i = get_global_id(0) * INTERNAL_BLOCK_SIZE;
    const ulong i_linear = get_group_id(0) * get_local_size(0) * INTERNAL_BLOCK_SIZE + get_local_id(0);
    const ulong j = get_global_id(1) * INTERNAL_BLOCK_SIZE;
    const ulong j_cached_idx_linear = get_group_id(1) * get_local_size(1) * INTERNAL_BLOCK_SIZE + get_local_id(0);

    __local real_type data_cache_i[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];
    __local real_type data_cache_j[FEATURE_BLOCK_SIZE][INTERNAL_BLOCK_SIZE * THREAD_BLOCK_SIZE];

    if (get_group_id(0) >= get_group_id(1)) {
        real_type temp[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };

        for (ulong dim = 0; dim < num_features; dim += FEATURE_BLOCK_SIZE) {
            // load data into shared memory
            for (uint internal = 0; internal < INTERNAL_BLOCK_SIZE; ++internal) {
                const ulong global_i = i_linear + internal * THREAD_BLOCK_SIZE;
                const ulong global_j = j_cached_idx_linear + internal * THREAD_BLOCK_SIZE;

                data_cache_i[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_i[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_i];
                data_cache_j[get_local_id(1)][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1)) * (num_rows + 1 + PADDING_SIZE) + global_j];
                data_cache_j[get_local_id(1) + THREAD_BLOCK_SIZE][internal * THREAD_BLOCK_SIZE + get_local_id(0)] = data_d[(dim + get_local_id(1) + THREAD_BLOCK_SIZE) * (num_rows + 1 + PADDING_SIZE) + global_j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // calculation
            for (uint block_dim = 0; block_dim < FEATURE_BLOCK_SIZE; ++block_dim) {
                for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
                    for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                        const real_type d = data_cache_i[block_dim][get_local_id(0) * INTERNAL_BLOCK_SIZE + internal_i] - data_cache_j[block_dim][get_local_id(1) * INTERNAL_BLOCK_SIZE + internal_j];
                        temp[internal_i][internal_j] += d * d;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for (uint internal_i = 0; internal_i < INTERNAL_BLOCK_SIZE; ++internal_i) {
            for (uint internal_j = 0; internal_j < INTERNAL_BLOCK_SIZE; ++internal_j) {
                const ulong global_i = i + internal_i;
                const ulong global_j = j + internal_j;

                if (global_i < num_rows && global_j < num_rows && global_i >= global_j) {
                    real_type temp_ij = temp[internal_i][internal_j];
                    temp_ij = exp(-gamma * temp_ij) + QA_cost - q[global_i] - q[global_j];
                    if (global_i == global_j) {
                        temp_ij += cost;
                    }

#ifdef PLSSVM_USE_GEMM
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i] = temp_ij;
                    ret[global_i * (num_rows + PADDING_SIZE) + global_j] = temp_ij;
#else
                    ret[global_j * (num_rows + PADDING_SIZE) + global_i - global_j * (global_j + 1) / 2] = temp_ij;
#endif
                }
            }
        }
    }
}