__kernel void kernel_q(__global double *q, __global  double *data_d, __global double *datlast, const int Ncols, const int Nrows){
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	int index = get_global_id(0); 
		double temp = 0;
		for(int i = 0; i < Ncols ; ++i){
			 temp += data_d[i * Nrows + index] * datlast[i];
			// printf("index: %i, temp: %f\n", index, data_d[i * Nrows + index]);
		}
		q[index] = temp;  
		
}