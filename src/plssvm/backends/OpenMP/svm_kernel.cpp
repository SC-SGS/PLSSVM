#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type, plssvm::kernel_function

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::openmp {

template <kernel_type kernel, typename real_type, typename... Args>
void device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const int add, Args &&...args) {
    using size_type = std::size_t;

    constexpr size_type BLOCK_SIZE = 64;

    const size_type dept = d.size();

#pragma omp parallel for collapse(2)
    for (size_type i = 0; i < dept; i += BLOCK_SIZE) {
        for (size_type j = 0; j < dept; j += BLOCK_SIZE) {
            for (size_type ii = 0; ii < BLOCK_SIZE && ii + i < dept; ++ii) {
                real_type ret_iii = 0.0;
                for (size_type jj = 0; jj < BLOCK_SIZE && jj + j < dept; ++jj) {
                    if (ii + i >= jj + j) {
                        const real_type temp = (kernel_function<kernel>(data[ii + i], data[jj + j], std::forward<Args>(args)...) + QA_cost - q[ii + i] - q[jj + j]) * add;
                        if (ii + i == jj + j) {
                            ret_iii += (temp + cost * add) * d[ii + i];
                        } else {
                            ret_iii += temp * d[jj + j];
#pragma omp atomic
                            ret[jj + j] += temp * d[ii + i];
                        }
                    }
                }
#pragma omp atomic
                ret[ii + i] += ret_iii;
            }
        }
    }
}

template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const int add) {
    device_kernel<kernel_type::linear>(q, ret, d, data, QA_cost, cost, add);
}
template void device_kernel_linear(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const int);
template void device_kernel_linear(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const int);

template <typename real_type>
void device_kernel_poly(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const int add, const int degree, const real_type gamma, const real_type coef0) {
    device_kernel<kernel_type::polynomial>(q, ret, d, data, QA_cost, cost, add, degree, gamma, coef0);
}
template void device_kernel_poly(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const int, const int, const float, const float);
template void device_kernel_poly(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const int, const int, const double, const double);

template <typename real_type>
void device_kernel_radial(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, const real_type QA_cost, const real_type cost, const int add, const real_type gamma) {
    device_kernel<kernel_type::rbf>(q, ret, d, data, QA_cost, cost, add, gamma);
}
template void device_kernel_radial(const std::vector<float> &, std::vector<float> &, const std::vector<float> &, const std::vector<std::vector<float>> &, const float, const float, const int, const float);
template void device_kernel_radial(const std::vector<double> &, std::vector<double> &, const std::vector<double> &, const std::vector<std::vector<double>> &, const double, const double, const int, const double);

}  // namespace plssvm::openmp

// TODO: look at further optimizations
// void kernel_linear(std::tuple<int,int> block, std::tuple<int,int> blockDim,real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add){
// 	int blockDimx = std::get<0>(blockDim);
// 	int blockDimy = std::get<1>(blockDim);
// 	for(int blockIdxx = 0; blockIdxx < std::get<0>(block); ++blockIdxx){
// 		for(int blockIdxy = 0; blockIdxy < std::get<1>(block); ++blockIdxy){
// 			for(int threadIdxx = 0; threadIdxx < blockDimy; ++ threadIdxx){
// 				for(int threadIdxy = 0; threadIdxy < blockDimy; ++ threadIdxy){
//
// 					int i =  blockIdxx * blockDimx * INTERNAL_BLOCK_SIZE;
// 					int j = blockIdxy * blockDimy * INTERNAL_BLOCK_SIZE;
//
// 					/*__shared__*/ real_t data_intern_i [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 					/*__shared__*/ real_t data_intern_j [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 					real_t matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = {};
// 					real_t data_j[INTERNAL_BLOCK_SIZE];
//
// 					if(i >= j){
// 						i += threadIdxx * INTERNAL_BLOCK_SIZE;
// 						const int ji = j +  threadIdxx * INTERNAL_BLOCK_SIZE;
// 						j += threadIdxy * INTERNAL_BLOCK_SIZE;
// 						for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 							{
// 								#pragma unroll(INTERNAL_BLOCK_SIZE)
// 								for(int block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id){
// 									const int data_index = vec_index + block_id;
// 									if(threadIdxy == block_id ) data_intern_i[threadIdxx][block_id] = data_d[data_index + i ];
// 									if(threadIdxy == block_id * 2 ) data_intern_j[threadIdxx][block_id] = data_d[data_index + ji];
// 								}
//
// 							}
// 							//__syncthreads();
//
// 							#pragma unroll(INTERNAL_BLOCK_SIZE)
// 							for(int data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index){
// 								data_j[data_index] = data_intern_j[threadIdxy][data_index];
// 							}
// 							//__syncthreads();
// 							#pragma unroll(INTERNAL_BLOCK_SIZE)
// 							for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 								const real_t data_i = data_intern_i[threadIdxx][x];
// 								#pragma unroll(INTERNAL_BLOCK_SIZE)
// 								for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 									matr[x][y] += data_i * data_j[y];
// 								}
// 							}
// 						}
// 						#pragma unroll(INTERNAL_BLOCK_SIZE)
// 						for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 							#pragma unroll(INTERNAL_BLOCK_SIZE)
// 							for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 								const real_t temp = (matr[x][y]  + QA_cost - q[i + x] - q[j + y]) * add;
// 								if(i + x > j + y){
// 									//atomicAdd(&ret[i + x], temp * d[j + y]);
// 									ret[i + x] += temp * d[j + y];
// 									//atomicAdd(&ret[j + y], temp * d[i + x]);
// 									ret[j + y] += temp * d[i + x];
// 								}else if(i + x == j + y){
// 									//atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
// 									ret[j + y]+= (temp + cost * add) * d[i + x];
// 								}
// 							}
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }
//
// void kernel_poly(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const int degree){
// 	int i =  blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
// 	int j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;
//
// 	/*__shared__*/ real_t data_intern_i [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 	/*__shared__*/ real_t data_intern_j [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 	real_t matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = {};
// 	real_t data_j[INTERNAL_BLOCK_SIZE];
//
// 	if(i >= j){
// 		i += threadIdx.x * INTERNAL_BLOCK_SIZE;
// 		const int ji = j +  threadIdx.x * INTERNAL_BLOCK_SIZE;
// 		j += threadIdx.y * INTERNAL_BLOCK_SIZE;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(INTERNAL_BLOCK_SIZE)
// 				for(int block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}
//
// 			}
// 			//__syncthreads();
//
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			//__syncthreads();
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(INTERNAL_BLOCK_SIZE)
// 				for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 					matr[x][y] += data_i * data_j[y];
// 				}
// 			}
// 		}
// 		#pragma unroll(INTERNAL_BLOCK_SIZE)
// 		for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 				const real_t temp = (pow(gamma * matr[x][y] + coef0, degree) + QA_cost - q[i + x] - q[j + y]) * add;
// 				if(i + x > j + y){
// 					atomicAdd(&ret[i + x], temp * d[j + y]);
// 					atomicAdd(&ret[j + y], temp * d[i + x]);
// 				}else if(i + x == j + y){
// 					atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
// 				}
// 			}
// 		}
// 	}
// }
//
// void kernel_radial(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma){
// 	int i =  blockIdx.x * blockDim.x * INTERNAL_BLOCK_SIZE;
// 	int j = blockIdx.y * blockDim.y * INTERNAL_BLOCK_SIZE;
//
// 	/*__shared__*/ real_t data_intern_i [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 	/*__shared__*/ real_t data_intern_j [THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
// 	real_t matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = {};
// 	real_t data_j[INTERNAL_BLOCK_SIZE];
//
// 	if(i >= j){
// 		i += threadIdx.x * INTERNAL_BLOCK_SIZE;
// 		const int ji = j +  threadIdx.x * INTERNAL_BLOCK_SIZE;
// 		j += threadIdx.y * INTERNAL_BLOCK_SIZE;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(INTERNAL_BLOCK_SIZE)
// 				for(int block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}
//
// 			}
// 			__syncthreads();
//
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			__syncthreads();
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(INTERNAL_BLOCK_SIZE)
// 				for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 					matr[x][y] += (data_i - data_j[y]) * (data_i - data_j[y]) ;
// 				}
// 			}
// 		}
//
// 		#pragma unroll(INTERNAL_BLOCK_SIZE)
// 		for(int x = 0; x < INTERNAL_BLOCK_SIZE; ++x){
// 			#pragma unroll(INTERNAL_BLOCK_SIZE)
// 			for(int y = 0; y < INTERNAL_BLOCK_SIZE; ++y){
// 				const real_t temp = (exp(-gamma * matr[x][y]) + QA_cost - q[i + x] - q[j + y]) * add;
// 				if(i + x > j + y){
// 					atomicAdd(&ret[i + x], temp * d[j + y]);
// 					atomicAdd(&ret[j + y], temp * d[i + x]);
// 				}else if(i + x == j + y){
// 					atomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
// 				}
// 			}
// 		}
// 	}
// }
