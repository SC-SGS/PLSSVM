__kernel void device_kernel_q_linear(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const int num_rows, const int first_feature, const int last_feature) {
    size_type index = get_global_id(0);
    real_type temp = 0.0;
    for (int i = first_feature; i < last_feature; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = temp;
}

__kernel void device_kernel_q_poly(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const int num_rows, const int num_cols, const int degree, const real_type gamma, const real_type coef0) {
    size_type index = get_global_id(0);
    real_type temp = 0.0;
    for (int i = 0; i < num_cols; ++i) {
        temp += data_d[i * num_rows + index] * data_last[i];
    }
    q[index] = pow(gamma * temp + coef0, degree);
}

__kernel void device_kernel_q_radial(__global real_type *q, __global real_type *data_d, __global real_type *data_last, const int num_rows, const int num_cols, const real_type gamma) {
    size_type index = get_global_id(0);
    real_type temp = 0.0;
    for (int i = 0; i < num_cols; ++i) {
        temp += (data_d[i * num_rows + index] - data_last[i]) * (data_d[i * num_rows + index] - data_last[i]);
    }
    q[index] = exp(-gamma * temp);
}