__global__ void kernel_q(double *q, double *data_d, double *datlast,const int Ncols, const int Nrows){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
		double temp = 0;
		for(int i = 0; i < Ncols ; ++i){
			 temp += data_d[i * Nrows + index] * datlast[i];
		}
		q[index] = temp;  
}