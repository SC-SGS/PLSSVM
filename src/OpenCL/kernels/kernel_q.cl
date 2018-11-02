__kernel void kernel_q(__global float *q, __global  float *data_d, __global float *datlast, const int Ncols, const int Nrows){
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = get_global_id(0); 
		float temp = 0;
		for(int i = 0; i < Ncols ; ++i){
			 temp += data_d[i * Nrows + index] * datlast[i];
			// printf("index: %i, temp: %f\n", index, data_d[i * Nrows + index]);
		}
		q[index] = temp;  
		
}