/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

// TODO: include?

//TODO: remove copy paste
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
static inline void __attribute__((overloadable)) AtomicAdd(__global const double *source, const double delta) {
    union {
        double f;
        ulong i;
    } oldVal;
    union {
        double f;
        ulong i;
    } newVal;
    do {
        oldVal.f = *source;
        newVal.f = oldVal.f + delta;
        // ++i;
    } while (atom_cmpxchg((volatile __global ulong *) source, oldVal.i, newVal.i) != oldVal.i);
}

static inline void __attribute__((overloadable)) AtomicAdd(__global const float *source, const float delta) {
    union {
        float f;
        unsigned i;
    } oldVal;
    union {
        float f;
        unsigned i;
    } newVal;
    do {
        oldVal.f = *source;
        newVal.f = oldVal.f + delta;
    } while (atom_cmpxchg((volatile __global unsigned *) source, oldVal.i, newVal.i) != oldVal.i);
}

__kernel void kernel_w(__global real_type *w_d, __global real_type *data_d, __global real_type *data_last_d, __global real_type *alpha_d, const size_type num_data_points, const size_type num_features) {
    size_type index = get_global_id(0);
    real_type temp = 0.0;
    if (index < num_features) {
        for (int dat = 0; dat < num_data_points - 1; ++dat) {
            temp += alpha_d[dat] * data_d[dat + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * index];
        }
        temp += alpha_d[num_data_points - 1] * data_last_d[index];
        w_d[index] = temp;
    }
}

__kernel void predict_points_poly(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const size_type num_data_points, __global const real_type *points, const size_type num_predict_points, const size_type num_features, const int degree, const real_type gamma, const real_type coef0) {
    const int data_point_index = get_global_id(0);
    const int predict_point_index = get_global_id(1);

    real_type temp = 0;
    if (predict_point_index < num_predict_points) {
        for (int feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += data_last_d[feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            } else {
                temp += data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] * points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index];
            }
        }

        temp = alpha_d[data_point_index] * pow(gamma * temp + coef0, degree);

        AtomicAdd(&out_d[predict_point_index], temp);
    }
}

__kernel void predict_points_rbf(__global real_type *out_d, __global const real_type *data_d, __global const real_type *data_last_d, __global const real_type *alpha_d, const size_type num_data_points, __global const real_type *points, const size_type num_predict_points, const size_type num_features, const real_type gamma) {
    const int data_point_index = get_global_id(0);
    const int predict_point_index = get_global_id(1);

    real_type temp = 0;
    if (predict_point_index < num_predict_points) {
        for (int feature_index = 0; feature_index < num_features; ++feature_index) {
            if (data_point_index == num_data_points) {
                temp += (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_last_d[feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            } else {
                temp += (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]) * (data_d[data_point_index + (num_data_points - 1 + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index] - points[predict_point_index + (num_predict_points + THREAD_BLOCK_SIZE * INTERNAL_BLOCK_SIZE) * feature_index]);
            }
        }

        temp = alpha_d[data_point_index] * exp(-gamma * temp);

        AtomicAdd(&out_d[predict_point_index], temp);
    }
}