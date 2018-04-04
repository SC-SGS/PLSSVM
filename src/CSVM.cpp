#include "CSVM.hpp"
#include <chrono>
#include <omp.h>  

CSVM::CSVM(double cost_, double epsilon_, unsigned kernel_, double degree_, double gamma_, double coef0_ , bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_){}

void CSVM::learn()
{
	vector<double> q;
	vector<double> b = value;
#pragma omp parallel sections
	{
#pragma omp section // generate q
		{
			q.reserve(data.size());
			for (int i = 0; i < data.size() - 1; ++i) {
				q.emplace_back(kernel_function(data.back(), data[i]));
			}
		}
#pragma omp section // generate right side from eguation
		{
			b.pop_back();
			b -= value.back();
		}
#pragma omp section // generate botom right from A
		{
			QA_cost = kernel_function(data.back(), data.back()) + 1 / cost;
		}
	}
	
	std::cout << "start CG" << std::endl;
	//solve minimization
    alpha = CG(b,1000,epsilon);


    alpha.emplace_back(-sum(alpha));
	bias = value.back() - QA_cost * alpha.back() - (q * alpha);
}



double CSVM::kernel_function(vector<double>& xi, vector<double>& xj){
	switch(kernel){
		case 0: return xi * xj;
		case 1: return std::pow(gamma * (xi*xj) + coef0 ,degree);
		case 2: return exp(-gamma * (xi-xj)*(xi-xj));
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
	
}

double CSVM::kernel_function(double* xi, double* xj, int dim)
{
	switch(kernel){
		case 0: return  mult(xi, xj, dim);
		case 1: return std::pow(gamma *  mult(xi, xj, dim) + coef0 ,degree);
		case 2: {double temp = 0;
			for(int i = 0; i < dim; ++i){
				temp += (xi[i]-xj[i]);
			}
			return exp(-gamma * temp * temp);}
		default: throw std::runtime_error("Can not decide wich kernel!");
	}
}



void CSVM::learn(string &filename, string &output_filename) {
	auto begin_parse = std::chrono::high_resolution_clock::now();
	if(filename.size() > 5 && endsWith(filename,  ".arff")){
		arffParser(filename);
	}else{
		libsvmParser(filename);
	}
	auto end_parse = std::chrono::high_resolution_clock::now();
	if(info){std::clog << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data  <<" in " << std::chrono::duration_cast<std::chrono::milliseconds>(end_parse - begin_parse).count() << " ms eingelesen" << std::endl << std::endl ;}

	learn();
	auto end_learn = std::chrono::high_resolution_clock::now();
    if(info) std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_parse).count() << " ms gelernt" << std::endl;

	writeModel(output_filename);
	auto end_write = std::chrono::high_resolution_clock::now();
    if(info){std::clog << std::endl << data.size()<<" Datenpunkte mit Dimension "<< Nfeatures_data <<" in " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << " geschrieben" << std::endl;
    }else if(times){
		std::clog << data.size()<<", "<< Nfeatures_data  <<", " << 0 << ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_learn - end_parse).count() << ", " <<std::chrono::duration_cast<std::chrono::milliseconds>(end_write-end_learn).count() << std::endl;
	} 
}


std::vector<double> CSVM::CG(const std::vector<double> &b, const int imax, const double eps)
{
	std::vector<double> x(b.size(), 1);
	double* datlast = &data.back()[0];
	static const size_t dim = data.back().size();
	static const size_t dept = b.size();
	
	//r = b - (A * x)
	///r = b;
	double r[dept];
	std::copy(b.begin(), b.end(), r);
	
	int bloksize = 64;

#pragma omp parallel for collapse(2) schedule(dynamic,8)
	for (int i = 1; i < (dept + bloksize); i = i + bloksize) {		
		for (int j = 0; j < (dept + bloksize); j = j + bloksize) {
			
			for(int ii = 0; ii < bloksize && ii + i< dept; ++ii){
				for(int jj = 0 ; jj < bloksize && jj + j < dept; ++jj){
					if(ii + i > jj + j ){
						double temp = kernel_function(&data[ii + i][0], &data[jj + j][0], dim) - kernel_function(datlast, &data[ii + i][0], dim);
						#pragma omp atomic
						r[jj + j] -= temp;
						#pragma omp atomic
						r[ii + i] -= temp; 
					}
				}
			}	
		}
	}


#pragma omp parallel for schedule(dynamic,8)
	for(int i = 0; i < dept; ++i){
		double kernel_dat_and_cost =  kernel_function(datlast, &data[i][0], dim) - QA_cost;
		#pragma omp atomic
		r[i] -= kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1/cost - (b.size()-i) * kernel_dat_and_cost;
		for(int j = i + 1; j < dept; ++j){
			#pragma omp atomic
			r[j] = r[j] + kernel_dat_and_cost;			
		}
	}

	
	std::cout << "r= b-Ax" <<std::endl;
	double d[b.size()] = {0};


	std::memcpy(d,r,dept*sizeof(double));

	
	double delta = mult(r, r, sizeof(r)/sizeof(double));	
	const double delta0 = delta;
	double alpha, beta;


	for(int run = 0; run < imax ; ++run){
	std::cout << "Start Iteration: " << run << std::endl;
	//Ad = A * d
	double Ad[dept] = {0.0} ;
		
#pragma omp parallel for collapse(2) schedule(dynamic,8)
	for (int i = 0; i < b.size(); i += bloksize) {		
		for (int j = 0; j < b.size(); j += bloksize) {

			double temp_data_i[bloksize][data[0].size()];
			double temp_data_j[bloksize][data[0].size()];
			for(int ii = 0; ii < bloksize; ++ii){
				
				if(ii + i< b.size())std::copy(data[ii + i].begin(), data[ii + i].end(), temp_data_i[ii]);
				if(ii + j< b.size())std::copy(data[ii + j].begin(), data[ii + j].end(), temp_data_j[ii]);
			}
			for(int ii = 0; ii < bloksize && ii + i< b.size(); ++ii){
				for(int jj = 0 ; jj < bloksize && jj + j < b.size(); ++jj){

					if(ii + i > jj + j ){
						double temp = kernel_function(temp_data_i[ii], temp_data_j[jj], dim) - kernel_function(datlast, temp_data_j[jj], dim);
						#pragma omp atomic
						Ad[jj + j] += temp * d[ii + i];
						#pragma omp atomic
						Ad[ii + i] += temp * d[jj + j]; 
					}
				}
				
			}	
		}
	}
	
		
#pragma omp parallel for schedule(dynamic,8)
	for(int i = 0; i < b.size(); ++i){
		double kernel_dat_and_cost =  kernel_function(datlast, &data[i][0], dim) - QA_cost;
		#pragma omp atomic
		Ad[i] +=  (kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1/cost - kernel_dat_and_cost) * d[i];
		for(int j = 0; j < i; ++j){
			#pragma omp atomic
			Ad[j] -= kernel_dat_and_cost * d[i];
			#pragma omp atomic
			Ad[i] -= kernel_dat_and_cost * d[j];
		}
	}


	alpha = delta / mult(d , Ad,  sizeof(d)/sizeof(double));
	x +=  mult(alpha, d,  sizeof(d)/sizeof(double));
	//r = b - (A * x)
	///r = b;
	std::copy(b.begin(), b.end(), r);

			
#pragma omp parallel for collapse(2) schedule(dynamic,8)
	for (int i = 0; i < (b.size() + bloksize); i += bloksize) {		
		for (int j = 0; j < (b.size() + bloksize); j += bloksize) {
			
			for(int ii = 0; ii < bloksize && ii + i< b.size(); ++ii){
				for(int jj = 0 ; jj < bloksize && jj + j < b.size(); ++jj){
					if(ii + i > jj + j ){
						double temp = kernel_function(&data[ii + i][0], &data[jj + j][0], dim) - kernel_function(datlast, &data[jj + j][0], dim);
						#pragma omp atomic
						r[jj + j] -= temp * x[ii + i];
						#pragma omp atomic
						r[ii + i] -= temp * x[jj + j]; 
					}
				}
			}	
		}
	}

#pragma omp parallel for schedule(dynamic,8)
	for(int i = 0; i < b.size(); ++i){
		double kernel_dat_and_cost =  kernel_function(datlast, &data[i][0], dim) - QA_cost;
		#pragma omp atomic
		r[i] -=  (kernel_function(&data[i][0], &data[i][0], dim) - kernel_function(datlast, &data[i][0], dim) + 1/cost - kernel_dat_and_cost) * x[i];
		for(int j = 0; j < i; ++j){
			#pragma omp atomic
			r[j] += kernel_dat_and_cost * x[i];
			#pragma omp atomic
			r[i] += kernel_dat_and_cost * x[j];
		}
	}

	
	delta = mult(r , r, b.size());
	//break;
	if(delta < eps * eps * delta0) break;
	beta = -mult(r, Ad, b.size()) / mult(d, Ad,b.size());
	add(mult(beta, d,  sizeof(d)/sizeof(double)),r, d, sizeof(d)/sizeof(double) );
	}
	
	return x;
}
	
