#include "plssvm/backends/OpenMP/svm-kernel.hpp"

#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm {

template <typename real_type>
void device_kernel_linear(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &ret, const std::vector<real_type> &d, const real_type QA_cost, const real_type cost, const int sign) {
    using size_type = std::size_t;
    constexpr size_type BLOCK_SIZE = 64;  // TODO: ?

    const std::vector<real_type> &data_last = data.back();
    const size_type dept = d.size();

    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (size_type i = 0; i < dept; i += BLOCK_SIZE) {
        for (size_type j = 0; j < dept; j += BLOCK_SIZE) {
            for (size_type ii = 0; ii < BLOCK_SIZE && ii + i < dept; ++ii) {
                for (size_type jj = 0; jj < BLOCK_SIZE && jj + j < dept; ++jj) {
                    if (ii + i > jj + j) {
                        const real_type temp = kernel_function<kernel_type::linear>(data[ii + i], data[jj + j])
                                               - kernel_function<kernel_type::linear>(data_last, data[jj + j]);
                        #pragma omp atomic
                        ret[jj + j] += temp * d[ii + i] * sign;
                        #pragma omp atomic
                        ret[ii + i] += temp * d[jj + j] * sign;
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (size_type i = 0; i < dept; ++i) {
        const real_type kernel_dat_and_cost = kernel_function<kernel_type::linear>(data_last, data[i]) - QA_cost;
        #pragma omp atomic
        ret[i] += (kernel_function<kernel_type::linear>(data[i], data[i]) - kernel_function<kernel_type::linear>(data_last, data[i]) + cost - kernel_dat_and_cost) * d[i] * sign;
        for (size_type j = 0; j < i; ++j) {
            #pragma omp atomic
            ret[j] -= kernel_dat_and_cost * sign * d[i];
            #pragma omp atomic
            ret[i] -= kernel_dat_and_cost * sign * d[j];
        }
    }
}

template void device_kernel_linear(const std::vector<std::vector<float>> &, std::vector<float> &, const std::vector<float> &, const float, const float, const int);
template void device_kernel_linear(const std::vector<std::vector<double>> &, std::vector<double> &, const std::vector<double> &, const double, const double, const int);
}  // namespace plssvm

// TODO: look at further optimizations
// void kernel_linear(std::tuple<int,int> block, std::tuple<int,int> blockDim,real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add){
// 	int blockDimx = std::get<0>(blockDim);
// 	int blockDimy = std::get<1>(blockDim);
// 	for(int blockIdxx = 0; blockIdxx < std::get<0>(block); ++blockIdxx){
// 		for(int blockIdxy = 0; blockIdxy < std::get<1>(block); ++blockIdxy){
// 			for(int threadIdxx = 0; threadIdxx < blockDimy; ++ threadIdxx){
// 				for(int threadIdxy = 0; threadIdxy < blockDimy; ++ threadIdxy){
//
// 					int i =  blockIdxx * blockDimx * INTERNALBLOCK_SIZE;
// 					int j = blockIdxy * blockDimy * INTERNALBLOCK_SIZE;
//
// 					/*__shared__*/ real_t data_intern_i [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 					/*__shared__*/ real_t data_intern_j [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 					real_t matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
// 					real_t data_j[INTERNALBLOCK_SIZE];
//
// 					if(i >= j){
// 						i += threadIdxx * INTERNALBLOCK_SIZE;
// 						const int ji = j +  threadIdxx * INTERNALBLOCK_SIZE;
// 						j += threadIdxy * INTERNALBLOCK_SIZE;
// 						for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 							{
// 								#pragma unroll(INTERNALBLOCK_SIZE)
// 								for(int block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id){
// 									const int data_index = vec_index + block_id;
// 									if(threadIdxy == block_id ) data_intern_i[threadIdxx][block_id] = data_d[data_index + i ];
// 									if(threadIdxy == block_id * 2 ) data_intern_j[threadIdxx][block_id] = data_d[data_index + ji];
// 								}
//
// 							}
// 							//__syncthreads();
//
// 							#pragma unroll(INTERNALBLOCK_SIZE)
// 							for(int data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index){
// 								data_j[data_index] = data_intern_j[threadIdxy][data_index];
// 							}
// 							//__syncthreads();
// 							#pragma unroll(INTERNALBLOCK_SIZE)
// 							for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 								const real_t data_i = data_intern_i[threadIdxx][x];
// 								#pragma unroll(INTERNALBLOCK_SIZE)
// 								for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
// 									matr[x][y] += data_i * data_j[y];
// 								}
// 							}
// 						}
// 						#pragma unroll(INTERNALBLOCK_SIZE)
// 						for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 							#pragma unroll(INTERNALBLOCK_SIZE)
// 							for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
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
// void kernel_poly(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const real_t degree){
// 	int i =  blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
// 	int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;
//
// 	/*__shared__*/ real_t data_intern_i [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 	/*__shared__*/ real_t data_intern_j [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 	real_t matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
// 	real_t data_j[INTERNALBLOCK_SIZE];
//
// 	if(i >= j){
// 		i += threadIdx.x * INTERNALBLOCK_SIZE;
// 		const int ji = j +  threadIdx.x * INTERNALBLOCK_SIZE;
// 		j += threadIdx.y * INTERNALBLOCK_SIZE;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(INTERNALBLOCK_SIZE)
// 				for(int block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}
//
// 			}
// 			//__syncthreads();
//
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			//__syncthreads();
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(INTERNALBLOCK_SIZE)
// 				for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
// 					matr[x][y] += data_i * data_j[y];
// 				}
// 			}
// 		}
// 		#pragma unroll(INTERNALBLOCK_SIZE)
// 		for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
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
// 	int i =  blockIdx.x * blockDim.x * INTERNALBLOCK_SIZE;
// 	int j = blockIdx.y * blockDim.y * INTERNALBLOCK_SIZE;
//
// 	/*__shared__*/ real_t data_intern_i [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 	/*__shared__*/ real_t data_intern_j [THREADBLOCK_SIZE][INTERNALBLOCK_SIZE];
// 	real_t matr[INTERNALBLOCK_SIZE][INTERNALBLOCK_SIZE] = {};
// 	real_t data_j[INTERNALBLOCK_SIZE];
//
// 	if(i >= j){
// 		i += threadIdx.x * INTERNALBLOCK_SIZE;
// 		const int ji = j +  threadIdx.x * INTERNALBLOCK_SIZE;
// 		j += threadIdx.y * INTERNALBLOCK_SIZE;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(INTERNALBLOCK_SIZE)
// 				for(int block_id = 0; block_id < INTERNALBLOCK_SIZE; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}
//
// 			}
// 			__syncthreads();
//
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int data_index = 0; data_index < INTERNALBLOCK_SIZE; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			__syncthreads();
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(INTERNALBLOCK_SIZE)
// 				for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
// 					matr[x][y] += (data_i - data_j[y]) * (data_i - data_j[y]) ;
// 				}
// 			}
// 		}
//
// 		#pragma unroll(INTERNALBLOCK_SIZE)
// 		for(int x = 0; x < INTERNALBLOCK_SIZE; ++x){
// 			#pragma unroll(INTERNALBLOCK_SIZE)
// 			for(int y = 0; y < INTERNALBLOCK_SIZE; ++y){
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
