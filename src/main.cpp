#include "CPU_CSVM.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <stdlib.h> /* srand, rand */
#include <time.h>
using namespace std;
bool info = true;

void exit_with_help() {
    if (info) {
        std::cerr << "Usage: svm-train [options] training_set_file [model_file]\n";
        std::cerr << "options:\n";
        std::cerr << "-t kernel_type : set type of kernel function (default 0)\n";
        std::cerr << "	0 -- linear: u'*v\n";
        std::cerr << "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n";
        std::cerr << "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n";
        std::cerr << "-d degree : set degree in kernel function (default 3)\n";
        std::cerr << "-g gamma : set gamma in kernel function (default 1/num_features)\n";
        std::cerr << "-r coef0 : set coef0 in kernel function (default 0)\n";
        std::cerr << "-c cost : set the parameter C (default 1)\n";
        std::cerr << "-e epsilon : set tolerance of termination criterion (default 0.001)\n";
        std::cerr << "-q : quiet mode (no outputs)" << std::endl;
    }
    exit(1);
}

int main(int argc, char *argv[]) {
    int kernel_type = 0;
    real_t degree = 3;
    real_t gamma;
    real_t coef0 = 0;
    real_t cost = 1;
    real_t eps = 0.001;
    std::string input_file_name, model_file_name;
    int i = 0;
    for (i = 1; i < argc; i++) {
        if (argv[i][0] != '-')
            break;
        if (++i >= argc)
            exit_with_help();
        switch (argv[i - 1][1]) {
        case 't':
            kernel_type = atoi(argv[i]);
            break;
        case 'd':
            degree = atoi(argv[i]);
            break;
        case 'g':
            gamma = atof(argv[i]);
            if (gamma == 0) {
                std::cerr << "gamma = 0 is not allowed, it doesnt make any sense!" << std::endl;
                exit_with_help();
            }
            break;
        case 'r':
            coef0 = atof(argv[i]);
            break;
        case 'c':
            cost = atof(argv[i]);
            break;
        case 'e':
            eps = atof(argv[i]);
            break;
        case 'q':
            info = false;
            i--;
            break;
        default:
            std::cerr << "Unknown option: -" << argv[i - 1][1] << std::endl;
            exit_with_help();
        }
    }

    if (i >= argc)
        exit_with_help();

    input_file_name = argv[i];

    if (i < argc - 1)
        model_file_name = argv[i + 1];
    else {
        std::size_t found = input_file_name.find_last_of("/\\");
        model_file_name = input_file_name.substr(found + 1) + ".model";
    }

    try {
        CPU_CSVM svm(cost, eps, kernel_type, degree, gamma, coef0, info);
        svm.learn(input_file_name, model_file_name);

    } catch (std::exception &e) {
        std::cout << "error" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
