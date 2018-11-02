__kernel void add_mult(__global float* vec1,__global float* vec2, float value, int dim){
	//int id = blockIdx.x * blockDim.x + threadIdx.x;
	int id = get_global_id(0)
	if(id < dim){
		vec1[id] += vec2[id] * value;
	}
}