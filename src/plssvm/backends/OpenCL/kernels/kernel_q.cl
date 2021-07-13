__kernel void kernel_q_old(__global real_type *q, __global  real_type *data_d, __global real_type *datlast, const int Ncols, const int Nrows){
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = get_global_id(0);
    real_type temp = 0;
    for(int i = 0; i < Ncols ; ++i){
         temp += data_d[i * Nrows + index] * datlast[i];
        // printf("index: %i, temp: %f\n", index, data_d[i * Nrows + index]);
    }
    q[index] = temp;
}

__kernel void kernel_q(__global real_type *q, __global real_type *data_d, __global real_type *datlast, const int Nrows, const int start, const int end) {
	int index = get_global_id(0);
    real_type temp = 0;
    for(int i = start; i < end ; ++i){
         temp += data_d[i * Nrows + index] * datlast[i];
    }
    q[index] = temp;
}