__kernel void add_mult(__global double* vec1,__global double* vec2, double value, int dim){
	//int id = blockIdx.x * blockDim.x + threadIdx.x;
	int id = get_global_id(0)
	if(id < dim){
		vec1[id] += vec2[id] * value;
	}
}