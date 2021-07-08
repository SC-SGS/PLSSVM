#pragma once

#include "plssvm/backend_types.hpp"  // plssvm::backend_type
#include "plssvm/kernel_types.hpp"   // plssvm::kernel_type

#include "cxxopts.hpp"
#include "fmt/core.h"  // fmt::print

#include <cstddef>      // std::size_t
#include <cstdlib>      // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move

namespace plssvm {

template <typename T>
class parameter {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    using real_type = T;
    using size_type = std::size_t;

    explicit parameter(std::string input_filename) :
        input_filename{ std::move(input_filename) } {
        model_name_from_input();
    }

    parameter(int argc, char **argv) {
        cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
        options
            .positional_help("training_set_file [model_file]")
            .show_positional_help();
        options
            .set_width(150)
            .set_tab_expansion()
            // clang-format off
            .add_options()
                ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear,\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<kernel_type>()->default_value("0"))
                ("d,degree", "degree in kernel function", cxxopts::value<real_type>()->default_value("3"))
                ("g,gamma", "gamma in kernel function (default: 1/num_features)", cxxopts::value<real_type>())
                ("r,coef0", "coef0 in kernel function", cxxopts::value<real_type>()->default_value("0"))
                ("c,cost", "the parameter C", cxxopts::value<real_type>()->default_value("1"))
                ("e,epsilon", "tolerance of termination criterion", cxxopts::value<real_type>()->default_value("0.001"))
                ("b,backend", "chooses the backend openmp|cuda|opencl", cxxopts::value<backend_type>()->default_value("openmp"))
                ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info))
                ("h,help", "print this helper message", cxxopts::value<bool>())
                ("input", "", cxxopts::value<std::string>(), "training_set_file")
                ("model", "", cxxopts::value<std::string>(), "model_file");
        // clang-format on

        // parse command line options
        cxxopts::ParseResult result;
        try {
            options.parse_positional({ "input", "model" });
            result = options.parse(argc, argv);
        } catch (const std::exception &e) {
            fmt::print("{}\n{}\n", e.what(), options.help());
            std::exit(EXIT_FAILURE);
        }

        // print help message and exit
        if (result.count("help")) {
            fmt::print("{}", options.help());
            std::exit(EXIT_SUCCESS);
        }

        // parse kernel_type and cast the value to the respective enum
        kernel = result["kernel_type"].as<kernel_type>();

        // parse degree
        degree = result["degree"].as<real_type>();

        // parse gamma
        if (result.count("gamma")) {
            gamma = result["gamma"].as<real_type>();
            if (gamma == 0.0) {
                fmt::print(stderr, "gamma = 0.0 is not allowed, it doesnt make any sense!\n");
                fmt::print("{}", options.help());
                std::exit(EXIT_FAILURE);
            }
        } else {
            gamma = 0.0;
        }

        // parse coef0
        coef0 = result["coef0"].as<real_type>();

        // parse cost
        cost = result["cost"].as<real_type>();

        // parse epsilon
        epsilon = result["epsilon"].as<real_type>();

        // parse backend_type and cast the value to the respective enum
        backend = result["backend"].as<backend_type>();

        // parse input data filename
        if (!result.count("input")) {
            fmt::print(stderr, "Error missing input file!");
            fmt::print("{}", options.help());
            std::exit(EXIT_FAILURE);
        }
        input_filename = result["input"].as<std::string>();

        // parse output model filename
        if (result.count("model")) {
            model_filename = result["model"].as<std::string>();
        } else {
            model_name_from_input();
        }
    }

    kernel_type kernel = kernel_type::linear;
    real_type degree = 3.0;
    real_type gamma = 0.0;
    real_type coef0 = 0.0;
    real_type cost = 1.0;
    real_type epsilon = 0.001;
    backend_type backend = backend_type::openmp;
    bool print_info = true;

    std::string input_filename;
    std::string model_filename;

  private:
    void model_name_from_input() {
        std::size_t pos = input_filename.find_last_of("/\\");
        model_filename = input_filename.substr(pos + 1) + ".model";
    }
};

}  // namespace plssvm