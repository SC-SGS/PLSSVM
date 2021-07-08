#include <plssvm/backends/OpenMP/svm-kernel.hpp>

#include <plssvm/detail/operators.hpp>
// void kernel_linear(std::tuple<int,int> block, std::tuple<int,int> blockDim,real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add){
// 	int blockDimx = std::get<0>(blockDim);
// 	int blockDimy = std::get<1>(blockDim);
// 	for(int blockIdxx = 0; blockIdxx < std::get<0>(block); ++blockIdxx){
// 		for(int blockIdxy = 0; blockIdxy < std::get<1>(block); ++blockIdxy){
// 			for(int threadIdxx = 0; threadIdxx < blockDimy; ++ threadIdxx){
// 				for(int threadIdxy = 0; threadIdxy < blockDimy; ++ threadIdxy){

// 					int i =  blockIdxx * blockDimx * BLOCKING_SIZE_THREAD;
// 					int j = blockIdxy * blockDimy * BLOCKING_SIZE_THREAD;

// 					/*__shared__*/ real_t data_intern_i [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 					/*__shared__*/ real_t data_intern_j [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 					real_t matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
// 					real_t data_j[BLOCKING_SIZE_THREAD];

// 					if(i >= j){
// 						i += threadIdxx * BLOCKING_SIZE_THREAD;
// 						const int ji = j +  threadIdxx * BLOCKING_SIZE_THREAD;
// 						j += threadIdxy * BLOCKING_SIZE_THREAD;
// 						for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 							{
// 								#pragma unroll(BLOCKING_SIZE_THREAD)
// 								for(int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id){
// 									const int data_index = vec_index + block_id;
// 									if(threadIdxy == block_id ) data_intern_i[threadIdxx][block_id] = data_d[data_index + i ];
// 									if(threadIdxy == block_id * 2 ) data_intern_j[threadIdxx][block_id] = data_d[data_index + ji];
// 								}

// 							}
// 							//__syncthreads();

// 							#pragma unroll(BLOCKING_SIZE_THREAD)
// 							for(int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index){
// 								data_j[data_index] = data_intern_j[threadIdxy][data_index];
// 							}
// 							//__syncthreads();
// 							#pragma unroll(BLOCKING_SIZE_THREAD)
// 							for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 								const real_t data_i = data_intern_i[threadIdxx][x];
// 								#pragma unroll(BLOCKING_SIZE_THREAD)
// 								for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
// 									matr[x][y] += data_i * data_j[y];
// 								}
// 							}
// 						}
// 						#pragma unroll(BLOCKING_SIZE_THREAD)
// 						for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 							#pragma unroll(BLOCKING_SIZE_THREAD)
// 							for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
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

namespace plssvm {

template <typename T>
T kernel_function(T *xi, T *xj, const std::size_t dim) {  // TODO:
    switch (0) {
        case 0:
            return mult(xi, xj, dim);
        default:
            throw std::runtime_error{ "Can not decide wich kernel!" };
    }
}

constexpr int bloksize = 64;

template <typename T>
void kernel_linear(const std::vector<T> &b, std::vector<std::vector<T>> &data, T *datlast, const T *q, std::vector<T> &ret, const T *d, const std::size_t dim, const T QA_cost, const T cost, const int add) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int i = 0; i < b.size(); i += bloksize) {
        for (int j = 0; j < b.size(); j += bloksize) {
            T temp_data_i[bloksize][data[0].size()];  //TODO:
            T temp_data_j[bloksize][data[0].size()];  //TODO:
            for (int ii = 0; ii < bloksize; ++ii) {
                if (ii + i < b.size())
                    std::copy(data[ii + i].begin(), data[ii + i].end(), temp_data_i[ii]);
                if (ii + j < b.size())
                    std::copy(data[ii + j].begin(), data[ii + j].end(), temp_data_j[ii]);
            }
            for (int ii = 0; ii < bloksize && ii + i < b.size(); ++ii) {
                for (int jj = 0; jj < bloksize && jj + j < b.size(); ++jj) {
                    if (ii + i > jj + j) {
                        T temp = kernel_function(temp_data_i[ii], temp_data_j[jj], dim) - kernel_function(datlast, temp_data_j[jj], dim);
                        #pragma omp atomic
                        ret[jj + j] += temp * d[ii + i] * add;
                        #pragma omp atomic
                        ret[ii + i] += temp * d[jj + j] * add;
                    }
                }
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < b.size(); ++i) {
        real_t kernel_dat_and_cost = kernel_function(datlast, &data[i][0], dim) - QA_cost;
        #pragma omp atomic
        ret[i] += (kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + cost - kernel_dat_and_cost) * d[i] * add;
        for (int j = 0; j < i; ++j) {
            #pragma omp atomic
            ret[j] -= kernel_dat_and_cost * add * d[i];
            #pragma omp atomic
            ret[i] -= kernel_dat_and_cost * add * d[j];
        }
    }
}

template void kernel_linear(const std::vector<float> &, std::vector<std::vector<float>> &, float *, const float *, std::vector<float> &, const float *, const std::size_t, const float, const float, const int);
template void kernel_linear(const std::vector<double> &, std::vector<std::vector<double>> &, double *, const double *, std::vector<double> &, const double *, const std::size_t, const double, const double, const int);
}  // namespace plssvm

// void kernel_poly(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const real_t degree){
// 	int i =  blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
// 	int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;

// 	/*__shared__*/ real_t data_intern_i [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 	/*__shared__*/ real_t data_intern_j [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 	real_t matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
// 	real_t data_j[BLOCKING_SIZE_THREAD];

// 	if(i >= j){
// 		i += threadIdx.x * BLOCKING_SIZE_THREAD;
// 		const int ji = j +  threadIdx.x * BLOCKING_SIZE_THREAD;
// 		j += threadIdx.y * BLOCKING_SIZE_THREAD;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(BLOCKING_SIZE_THREAD)
// 				for(int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}

// 			}
// 			//__syncthreads();

// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			//__syncthreads();
// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(BLOCKING_SIZE_THREAD)
// 				for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
// 					matr[x][y] += data_i * data_j[y];
// 				}
// 			}
// 		}
// 		#pragma unroll(BLOCKING_SIZE_THREAD)
// 		for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
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

// void kernel_radial(real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma){
// 	int i =  blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
// 	int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;

// 	/*__shared__*/ real_t data_intern_i [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 	/*__shared__*/ real_t data_intern_j [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
// 	real_t matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
// 	real_t data_j[BLOCKING_SIZE_THREAD];

// 	if(i >= j){
// 		i += threadIdx.x * BLOCKING_SIZE_THREAD;
// 		const int ji = j +  threadIdx.x * BLOCKING_SIZE_THREAD;
// 		j += threadIdx.y * BLOCKING_SIZE_THREAD;
// 		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
// 			{
// 				#pragma unroll(BLOCKING_SIZE_THREAD)
// 				for(int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id){
// 					const int data_index = vec_index + block_id;
// 					if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];
// 					if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
// 				}

// 			}
// 			__syncthreads();

// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index){
// 				data_j[data_index] = data_intern_j[threadIdx.y][data_index];
// 			}
// 			__syncthreads();
// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 				const real_t data_i = data_intern_i[threadIdx.x][x];
// 				#pragma unroll(BLOCKING_SIZE_THREAD)
// 				for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
// 					matr[x][y] += (data_i - data_j[y]) * (data_i - data_j[y]) ;
// 				}
// 			}
// 		}

// 		#pragma unroll(BLOCKING_SIZE_THREAD)
// 		for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
// 			#pragma unroll(BLOCKING_SIZE_THREAD)
// 			for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
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
