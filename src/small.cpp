#include <iostream>
#include <vector>
#include "operators.hpp"
#include "CSVM.hpp"

int main(int argc, char* argv[])
{
	int kernel_type = 0;
	double degree = 3;
	double gamma;
	double coef0 = 0;
	double cost = 1;
	double eps = 0.001;
    std::string input_file_name("../25x25.txt");
    std::string model_file_name("../25x25.txt.model");
    try{
		CSVM svm(cost, eps, kernel_type, degree, gamma, coef0, true);

		svm.learn(input_file_name, model_file_name);
   
	}catch (std::exception &e) {
		std::cout << "error" << std::endl;
		std::cerr << e.what() <<std::endl;
	}

}