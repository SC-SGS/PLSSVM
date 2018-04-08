#include "CSVM.hpp"

__global__ void kernel_predict(double *data_d, double *w, int dim, double *out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0;
    for(int feature = 0; feature < dim ; ++feature){
        temp += w[feature] * data_d[index * dim + feature];
    }
    if(temp > 0) {
        out[index] = 1;
    }else{
        out[index] = -1;
    }
}

__global__ void kernel_w(double* w_d, double* data_d, double* alpha_d, int count ){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    double temp = 0;
    for(int dat = 0; dat < count ; ++dat){
        temp += alpha_d[index] * data_d[dat * count + index];
    }
    w_d[index] = temp;
}


std::vector<double> CSVM::predict(double *data, int dim , int count){
    double *data_d, *out;
    cudaMalloc((void **) &data_d, dim * count * sizeof(double));
    cudaMalloc((void **) &out, count * sizeof(double));
    cudaMemcpy(data_d, data, dim * count * sizeof(double), cudaMemcpyHostToDevice);

    kernel_predict<<<((int)count/1024) + 1,  std::min(count, 1024)>>>(data, w_d, dim, out);

    std::vector<double> ret(count);
    cudaDeviceSynchronize();
    cudaMemcpy(&ret[0], out, count * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(data_d);
    cudaFree(out);

    return ret;
}


void CSVM::load_w(){
    cudaMalloc((void **) &w_d, Nfeatures_data * sizeof(double));
    double *alpha_d;
    cudaMalloc((void **) &alpha_d, Nfeatures_data * sizeof(double));
    cudaMemcpy(alpha_d, &alpha[0], Nfeatures_data* sizeof(double), cudaMemcpyHostToDevice);

    kernel_w<<<((int)Nfeatures_data/1024) + 1,  std::min((int)Nfeatures_data, 1024)>>>(w_d, data_d, alpha_d, Ndatas_data);

    cudaDeviceSynchronize();
    cudaFree(alpha_d);

}
