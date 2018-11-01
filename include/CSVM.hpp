#ifndef CSVM_H
#define CSVM_H
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <omp.h> 
#include <sstream> 
#include <cstdlib>
#include <stdlib.h>
#include <utility>
#include <tuple>
#include <math.h>

#include "operators.hpp"

#include <tuple>


#ifdef WITH_OPENCL
#include "../src/OpenCL/manager/configuration.hpp"
#include "../src/OpenCL/manager/device.hpp"
#include "../src/OpenCL/manager/manager.hpp"
#include "DevicePtrOpenCL.hpp"
#include <stdexcept>
#endif

const bool times = 0;

static const unsigned CUDABLOCK_SIZE = 16;
static const int BLOCKING_SIZE_THREAD = 6;

// static const unsigned CUDABLOCK_SIZE = 7;
// static const int BLOCKING_SIZE_THREAD = 2;

class CSVM
{
    public:
        CSVM(double, double, unsigned, double, double, double, bool);
		void learn(std::string&, std::string&);
        
		const double& getB() const { return bias; };
        void load_w();
        std::vector<double> predict(double*, int, int);
    protected:

    private:
        const bool info; 
        double cost;
        const double epsilon;
        const unsigned kernel;
        const double degree;
        double gamma;
        const double coef0;
        double bias;
		double QA_cost;
        std::vector<std::vector<double> > data;
		size_t Nfeatures_data;
		size_t Ndatas_data;
        std::vector<double> value;
        std::vector<double> alpha;


        void learn();
		
        double kernel_function(std::vector<double>&, std::vector<double>&);
        double kernel_function(double*, double*, int);

        void libsvmParser(std::string&);
        void arffParser(std::string&);
        void writeModel(std::string&);

        void loadDataDevice();
		std::vector<double> CG(const std::vector<double> &b, const int , const double );


        #ifdef WITH_OPENCL
	        opencl::manager_t manager{"../platform_configuration.cfg"};
	        opencl::device_t first_device;
            cl_kernel kernel_q_cl;
            cl_kernel svm_kernel_linear;
            opencl::DevicePtrOpenCL<double>  datlast_cl;
            opencl::DevicePtrOpenCL<double>  data_cl;
        #endif

        #ifdef WITH_CUDA
            double *data_d;
            double *datlast;
            double *w_d;
        #endif
    };

#endif // C-SVM_H


