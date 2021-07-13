__kernel void add_mult(__global real_type* vec1,__global real_type* vec2, real_type value, int dim){
	//int id = blockIdx.x * blockDim.x + threadIdx.x;
	int id = get_global_id(0)
	if(id < dim) {
		vec1[id] += vec2[id] * value;
	}
}