#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

namespace plssvm {

template <typename real_type>
__global__ void kernel_predict(real_type *data_d, real_type *w, int dim, real_type *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    for (int feature = 0; feature < dim; ++feature) {
        temp += w[feature] * data_d[index * dim + feature];
    }
    if (temp > 0) {
        out[index] = 1;
    } else {
        out[index] = -1;
    }
}
template __global__ void kernel_predict(float *, float *, int, float *);
template __global__ void kernel_predict(double *, double *, int, double *);

template <typename real_type>
__global__ void kernel_w(real_type *w_d, real_type *data_d, real_type *alpha_d, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    for (int dat = 0; dat < count; ++dat) {
        temp += alpha_d[index] * data_d[dat * count + index];
    }
    w_d[index] = temp;
}
template __global__ void kernel_w(float *, float *, float *, int);
template __global__ void kernel_w(double *, double *, double *, int);

template <typename T>
auto CUDA_CSVM<T>::predict(const real_type *data, const size_type dim, const size_type count) -> std::vector<real_type> {
    real_type *data_d, *out;
    cudaMalloc((void **) &data_d, dim * count * sizeof(real_type));
    cudaMalloc((void **) &out, count * sizeof(real_type));
    cudaMemcpy(data_d, data, dim * count * sizeof(real_type), cudaMemcpyHostToDevice);

    kernel_predict<<<((int) count / 1024) + 1, std::min(count, 1024)>>>(data, w_d, dim, out);

    std::vector<real_type> ret(count);
    cudaDeviceSynchronize();
    cudaMemcpy(&ret[0], out, count * sizeof(real_type), cudaMemcpyDeviceToHost);
    cudaFree(data_d);
    cudaFree(out);

    return ret;
}

template <typename T>
void CUDA_CSVM<T>::load_w() {
    cudaMalloc((void **) &w_d, num_features_ * sizeof(real_type));
    real_type *alpha_d;
    cudaMalloc((void **) &alpha_d, num_features_ * sizeof(real_type));
    cudaMemcpy(alpha_d, &alpha_[0], num_features_ * sizeof(real_type), cudaMemcpyHostToDevice);

    // TODO:
    // kernel_w<<<((int)num_features/1024) + 1,  std::min((int)num_features, 1024)>>>(w_d, data_d, alpha_d, num_data_points);

    cudaDeviceSynchronize();
    cudaFree(alpha_d);
}

template class CUDA_CSVM<float>;
template class CUDA_CSVM<double>;

}  // namespace plssvm
