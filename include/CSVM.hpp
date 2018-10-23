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

const bool times = 0;

static const unsigned CUDABLOCK_SIZE = 16;
static const int BLOCKING_SIZE_THREAD = 6;

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
		double *data_d;
		double *datlast;
        double *w_d;
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
};

#endif // C-SVM_H


