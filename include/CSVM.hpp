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



// static const unsigned CUDABLOCK_SIZE = 7;
// static const unsigned BLOCKING_SIZE_THREAD = 2;

class CSVM
{
    public:
        CSVM(real_t, real_t, unsigned, real_t, real_t, real_t, bool);
		void learn(std::string&, std::string&);
        
		const real_t& getB() const { return bias; };
        void load_w();
        std::vector<real_t> predict(real_t*, int, int);
    protected:

    private:
        const bool info; 
        real_t cost;
        const real_t epsilon;
        const unsigned kernel;
        const real_t degree;
        real_t gamma;
        const real_t coef0;
        real_t bias;
		real_t QA_cost;
        std::vector<std::vector<real_t> > data;
		size_t Nfeatures_data;
		size_t Ndatas_data;
        std::vector<real_t> value;
        std::vector<real_t> alpha;


        void learn();
		
        inline real_t kernel_function(std::vector<real_t>&, std::vector<real_t>&);
        inline real_t kernel_function(real_t*, real_t*, int);

        void libsvmParser(std::string&);
        void arffParser(std::string&);
        void writeModel(std::string&);

        void loadDataDevice();

        

		std::vector<real_t> CG(const std::vector<real_t> &b, const int , const real_t );


        #ifdef WITH_OPENCL
            inline void resizeData(int boundary);
            inline void resizeData(const int device, int boundary);
            inline void resizeDatalast(int boundary);
            inline void resizeDatalast(const int device, int boundary);
            inline void resize(const int old_boundary,const int new_boundary);
	        opencl::manager_t manager{"../platform_configuration.cfg"};
	        opencl::device_t first_device;
            std::vector<cl_kernel> kernel_q_cl;
            std::vector<cl_kernel> svm_kernel_linear;
            std::vector<opencl::DevicePtrOpenCL<real_t> > datlast_cl;
            std::vector<opencl::DevicePtrOpenCL<real_t> > data_cl;
        #endif

        #ifdef WITH_CUDA
            real_t *data_d;
            real_t *datlast;
            real_t *w_d;
        #endif
    };

#endif // C-SVM_H


