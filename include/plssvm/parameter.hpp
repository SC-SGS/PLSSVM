/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements a class encapsulating all necessary parameters possibly provided through command line arguments.
 */

#pragma once

#include "plssvm/backend_types.hpp"  // plssvm::backend_type
#include "plssvm/kernel_types.hpp"   // plssvm::kernel_type

#include "cxxopts.hpp"  // command line parsing
#include "fmt/core.h"   // fmt::print, fmt::format

#include <algorithm>    // std::transform
#include <cctype>       // std::tolower
#include <cstddef>      // std::size_t
#include <cstdlib>      // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <utility>      // std::move

namespace plssvm {

/**
 * @brief Class encapsulating all necessary parameters possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    /// Unsigned integer type.
    using size_type = std::size_t;

    /**
     * @brief Set all parameters to their default values.
     * @param[in] input_filename the name of the data file
     */
    explicit parameter(std::string input_filename) :
        input_filename{ std::move(input_filename) } {
        model_name_from_input();
    }

    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the parameters accordingly.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parameter(int argc, char **argv) {
        // small helper function to convert a string to a lowercase string
        auto as_lowercase = [](const std::string_view str) -> std::string {
            std::string lowercase_str{ str };
            std::transform(str.begin(), str.end(), lowercase_str.begin(), [](const char c) { return std::tolower(c); });
            return lowercase_str;
        };

        cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
        options
            .positional_help("training_set_file [model_file]")
            .show_positional_help();
        options
            .set_width(150)
            .set_tab_expansion()
            // clang-format off
            .add_options()
                ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear,\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<kernel_type>()->default_value(std::to_string(detail::to_underlying(kernel))))
                ("d,degree", "degree in kernel function", cxxopts::value<real_type>()->default_value(fmt::format("{}", degree)))
                ("g,gamma", "gamma in kernel function (default: 1 / num_features)", cxxopts::value<real_type>())
                ("r,coef0", "coef0 in kernel function", cxxopts::value<real_type>()->default_value(fmt::format("{}", coef0)))
                ("c,cost", "the parameter C", cxxopts::value<real_type>()->default_value(fmt::format("{}", cost)))
                ("e,epsilon", "tolerance of termination criterion", cxxopts::value<real_type>()->default_value(fmt::format("{}", epsilon)))
                ("b,backend", "chooses the backend openmp|cuda|opencl", cxxopts::value<backend_type>()->default_value(as_lowercase(fmt::format("{}", backend))))
                ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info)->default_value(fmt::format("{}", !print_info)))
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

        // parse print info
        print_info = !print_info;

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

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    kernel_type kernel = kernel_type::linear;
    /// The degree parameter used in the polynomial kernel function.
    real_type degree = 3.0;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma = 0.0;
    /// The coef0 parameter used in the polynomial kernel function.
    real_type coef0 = 0.0;
    /// The cost parameter in the C-SVM.
    real_type cost = 1.0;
    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = 0.001;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    bool print_info = false;
    /// The used backend: OpenMP, OpenCL or CUDA.
    backend_type backend = backend_type::openmp;

    /// The name of the data file to parse.
    std::string input_filename;
    /// The name of the model file to write the learned Support Vectors to.
    std::string model_filename;

  private:
    /*
     * Generate model filename based on the name of the input file.
     */
    void model_name_from_input() {
        std::size_t pos = input_filename.find_last_of("/\\");
        model_filename = input_filename.substr(pos + 1) + ".model";
    }
};

}  // namespace plssvm