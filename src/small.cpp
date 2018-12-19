#include <iostream>
#include <vector>
#include "operators.hpp"
#include "CSVM.hpp"

int main(int argc, char* argv[])
{
	int kernel_type = 0;
	real_t degree = 3;
	real_t gamma;
	real_t coef0 = 0;
	real_t cost = 1;
	real_t eps = 0.001;
    // std::string input_file_name("../25x25.txt");
    // std::string model_file_name("../25x25.txt.model");
    // std::string input_file_name("../5000x2000.txt");
    // std::string model_file_name("../5000x2000.txt.model");
    // std::string model_file_name("../262144x4096.txt.model");
    // std::string input_file_name("../262144x4096.txt");
    // std::string model_file_name("../192x192.txt.model");
    // std::string input_file_name("../192x192.txt");
    std::string model_file_name("../90x90.txt.model");
    std::string input_file_name("../90x90.txt");
    try{
		CSVM svm(cost, eps, kernel_type, degree, gamma, coef0, true);

		svm.learn(input_file_name, model_file_name);
   
	}catch (std::exception &e) {
		std::cout << "error" << std::endl;
		std::cerr << e.what() <<std::endl;
	}

}