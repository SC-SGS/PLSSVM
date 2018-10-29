__kernel void init(__global double* vec, double value, int size){
	// int id = blockIdx.x * blockDim.x + threadIdx.x;
    int id = get_global_id(0);
	if(id < size) vec[id] = value;
}