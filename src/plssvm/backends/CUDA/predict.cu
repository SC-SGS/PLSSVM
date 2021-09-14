/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/CUDA/csvm.hpp"

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::detail::cuda::device_ptr

namespace plssvm::cuda {

template <typename real_type>
__global__ void kernel_predict(const real_type *data_d, const real_type *w, int dim, real_type *out) {
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
template __global__ void kernel_predict(const float *, const float *, int, float *);
template __global__ void kernel_predict(const double *, const double *, int, double *);

template <typename real_type>
__global__ void kernel_w(real_type *w_d, const real_type *data_d, const real_type *alpha_d, int count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_type temp = 0;
    for (int dat = 0; dat < count; ++dat) {
        temp += alpha_d[index] * data_d[dat * count + index];
    }
    w_d[index] = temp;
}
template __global__ void kernel_w(float *, const float *, const float *, int);
template __global__ void kernel_w(double *, const double *, const double *, int);

//template <typename T>
//auto csvm<T>::predict(const real_type *data, const size_type dim, const size_type count) -> std::vector<real_type> {
//    cuda::detail::device_ptr<real_type> data_d{ dim * count };
//    data_d.memcpy_to_device(data);
//    cuda::detail::device_ptr<real_type> out{ count };
//
//    kernel_predict<<<((int) count / 1024) + 1, std::min(count, static_cast<size_type>(1024))>>>(data_d.get(), w_d_.get(), dim, out.get());
//
//    std::vector<real_type> ret(count);
//    cuda::detail::device_synchronize();
//    out.memcpy_to_host(ret);
//
//    return ret;
//}

//template <typename T>
//void csvm<T>::load_w() {
//    w_d_ = cuda::detail::device_ptr<real_type>{ num_features_ };
//    cuda::detail::device_ptr<real_type> alpha_d{ num_features_ };
//    alpha_d.memcpy_to_device(alpha_, 0, num_features_);  // TODO: ????
//
//    // TODO:
//    // kernel_w<<<((int) num_features_ / 1024) + 1,  std::min((int) num_features_, 1024)>>>(w_d_.get(), data_d.get(), alpha_d.get(), num_data_points_);
//
//    cuda::detail::device_synchronize();
//}
//
//template class csvm<float>;
//template class csvm<double>;

}  // namespace plssvm::cuda
