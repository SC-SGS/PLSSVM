#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
void __attribute__((overloadable)) AtomicAdd(__global double *source, double delta) {
    union {
        double f;
        ulong i;
    } oldVal;
    union {
        double f;
        ulong i;
    } newVal;
    // int i = 0;
    do {
        oldVal.f = *source;
        newVal.f = oldVal.f + delta;
        // ++i;
    } while (atom_cmpxchg((volatile __global ulong *) source, oldVal.i, newVal.i) != oldVal.i);
    // if (i > 1) printf("%i\n", i);
}

void __attribute__((overloadable)) AtomicAdd(__global float *source, float delta) {
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

__kernel void kernel_linear(__global const real_type *q, __global real_type *ret, __global const real_type *d, __global const real_type *data_d, const real_type QA_cost, const real_type cost, const int Ncols, const int Nrows, const int add, const int start, const int end) {
    int i = get_group_id(0) * (get_local_size(0) * INTERNALBLOCK_SIZE);
    int j = get_group_id(1) * (get_local_size(1) * INTERNALBLOCK_SIZE);

    __private real_type matr = 0.0;

    if (i >= j) {
        i += get_local_id(0) * INTERNALBLOCK_SIZE;
        j += get_local_id(1) * INTERNALBLOCK_SIZE;
        #pragma unroll(INTERNALBLOCK_SIZE)
        for (int k = 0; k < INTERNALBLOCK_SIZE; ++k) {
            real_type ret_k = 0;
            #pragma unroll(INTERNALBLOCK_SIZE)
            for (int l = i; l < INTERNALBLOCK_SIZE + i; ++l) {
                matr = 0.0;
                for (int vec_index = 0; vec_index < Ncols * Nrows; vec_index += Nrows) {
                    matr += data_d[vec_index + l] * data_d[vec_index + j];
                    if (j == 1 && i == 1)
                        printf("%f, %f\n", data_d[vec_index + l], data_d[vec_index + j]);
                    // 	printf("%i\n", vec_index + l);
                    // printf("%i\n",  vec_index + j);
                }

                const real_type temp = (matr + QA_cost - q[l] - q[j]) * add;
                if (j == 1 && i == 1)
                    printf("%f, %f, %f, %i\n", matr, q[l], q[j], j);
                if (l > j) {
                    AtomicAdd(&ret[l], temp * d[j]);
                    ret_k += temp * d[l];
                    // if(k == 0 && i == 1) printf("%f, %f, %i\n", temp, d[l], l);
                } else if (l == j) {
                    ret_k += (temp + cost * add) * d[l];
                }
            }
            AtomicAdd(&ret[j], ret_k);
            //if(j == 1) printf("%f, %i\n", ret_k, i);
            j++;
        }
    }
}
