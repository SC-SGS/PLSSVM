#include "CSVM.hpp"

__global__ void kernel_predict(real_t *data_d, real_t *w, int dim, real_t *out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_t temp = 0;
    for(int feature = 0; feature < dim ; ++feature){
        temp += w[feature] * data_d[index * dim + feature];
    }
    if(temp > 0) {
        out[index] = 1;
    }else{
        out[index] = -1;
    }
}

__global__ void kernel_w(real_t* w_d, real_t* data_d, real_t* alpha_d, int count ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    real_t temp = 0;
    for(int dat = 0; dat < count ; ++dat){
        temp += alpha_d[index] * data_d[dat * count + index];
    }
    w_d[index] = temp;
}


std::vector<real_t> CSVM::predict(real_t *data, int dim , int count){
    real_t *data_d, *out;
    cudaMalloc((void **) &data_d, dim * count * sizeof(real_t));
    cudaMalloc((void **) &out, count * sizeof(real_t));
    cudaMemcpy(data_d, data, dim * count * sizeof(real_t), cudaMemcpyHostToDevice);

    kernel_predict<<<((int)count/1024) + 1,  std::min(count, 1024)>>>(data, w_d, dim, out);

    std::vector<real_t> ret(count);
    cudaDeviceSynchronize();
    cudaMemcpy(&ret[0], out, count * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaFree(data_d);
    cudaFree(out);

    return ret;
}


void CSVM::load_w(){
    cudaMalloc((void **) &w_d, Nfeatures_data * sizeof(real_t));
    real_t *alpha_d;
    cudaMalloc((void **) &alpha_d, Nfeatures_data * sizeof(real_t));
    cudaMemcpy(alpha_d, &alpha[0], Nfeatures_data* sizeof(real_t), cudaMemcpyHostToDevice);

    // TODO:
    // kernel_w<<<((int)Nfeatures_data/1024) + 1,  std::min((int)Nfeatures_data, 1024)>>>(w_d, data_d, alpha_d, Ndatas_data);

    cudaDeviceSynchronize();
    cudaFree(alpha_d);

}
