#include "CPU_CSVM.hpp"
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include <stdlib.h> /* srand, rand */
#include <time.h>

#include <cxxopts.hpp>

bool info;

int main(int argc, char *argv[]) {

    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("training_set_file [model_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear,\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<int>()->default_value("0")) //TODO: as enum and check if in range 0..2
            ("d,degree", "degree in kernel function", cxxopts::value<real_t>()->default_value("3"))
            ("g,gamma", "gamma in kernel function (default: 1/num_features)", cxxopts::value<real_t>())
            ("r,coef0", "coef0 in kernel function", cxxopts::value<real_t>()->default_value("0"))
            ("c,cost", "the parameter C", cxxopts::value<real_t>()->default_value("1"))
            ("e,epsilon", "tolerance of termination criterion", cxxopts::value<real_t>()->default_value("0.001"))
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(info))("h,help", "print this helper message", cxxopts::value<bool>())
            ("input", "", cxxopts::value<std::string>(), "training_set_file")("model", "", cxxopts::value<std::string>(), "model_file");
    // clang-format on

    cxxopts::ParseResult result;
    try {
        options.parse_positional({"input", "model"});
        result = options.parse(argc, argv);
    } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }
    info = !info;
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    int kernel_type = result["kernel_type"].as<int>();
    real_t gamma;
    if (result.count("gamma")) {
        gamma = result["gamma"].as<real_t>();
        if (gamma == 0) {
            std::cerr << "gamma = 0 is not allowed, it doesnt make any sense!" << std::endl;
            std::cout << options.help() << std::endl;
            exit(1);
        }
    } else {
        gamma = 0;
    }

    if (!result.count("input")) {
        std::cerr << "Error missing input file!!" << std::endl;
        std::cout << options.help() << std::endl;
        exit(1);
    }
    std::string input_file_name = result["input"].as<std::string>();
    std::string model_file_name;
    if (result.count("model")) {
        model_file_name = result["model"].as<std::string>();
    } else {
        std::size_t found = input_file_name.find_last_of("/\\");
        model_file_name = input_file_name.substr(found + 1) + ".model";
    }

    try {
        // std::unique_ptr<CSVM> svm = make_SVM(backend, result["cost"].as<real_t>(), result["epsilon"].as<real_t>(), kernel_type, degree = result["degree"].as<real_t>(), gamma, result["coef0"].as<real_t>(), info);
        CPU_CSVM svm(result["cost"].as<real_t>(), result["epsilon"].as<real_t>(), kernel_type, result["degree"].as<real_t>(), gamma, result["coef0"].as<real_t>(), info);
        svm.learn(input_file_name, model_file_name);

    } catch (std::exception &e) {
        std::cout << "error" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    return 0;
}
