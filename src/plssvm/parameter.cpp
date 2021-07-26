/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/parameter.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "cxxopts.hpp"  // command line parsing
#include "fmt/core.h"   // fmt::print, fmt::format

#include <algorithm>  // std::transform
#include <cctype>     // std::tolower
#include <cstddef>    // std::size_t
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <string>     // std::string
#include <utility>    // std::move

namespace plssvm {

template <typename T>
parameter<T>::parameter(std::string input_filename) :
    input_filename{ std::move(input_filename) } {
    model_name_from_input();
}

template <typename T>
parameter<T>::parameter(int argc, char **argv) {
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
            ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear: u'*v\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<kernel_type>()->default_value(fmt::format("{}", detail::to_underlying(kernel))))
            ("d,degree", "set degree in kernel function", cxxopts::value<real_type>()->default_value(fmt::format("{}", degree)))
            ("g,gamma", "set gamma in kernel function (default: 1 / num_features)", cxxopts::value<real_type>())
            ("r,coef0", "set coef0 in kernel function", cxxopts::value<real_type>()->default_value(fmt::format("{}", coef0)))
            ("c,cost", "set the parameter C", cxxopts::value<real_type>()->default_value(fmt::format("{}", cost)))
            ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<real_type>()->default_value(fmt::format("{}", epsilon)))
            ("b,backend", "chooses the backend openmp|cuda|opencl|sycl", cxxopts::value<backend_type>()->default_value(as_lowercase(fmt::format("{}", backend))))
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

template <typename T>
void parameter<T>::model_name_from_input() {
    std::size_t pos = input_filename.find_last_of("/\\");
    model_filename = input_filename.substr(pos + 1) + ".model";
}

// explicitly instantiate template class
template class parameter<float>;
template class parameter<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
    return out << fmt::format(
               "kernel_type     {}\n"
               "degree          {}\n"
               "gamma           {}\n"
               "coef0           {}\n"
               "cost            {}\n"
               "epsilon         {}\n"
               "print_info      {}\n"
               "backend         {}\n"
               "input_filename  {}\n"
               "model_filename  {}\n"
               "real_type       {}\n",
               params.kernel,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               params.epsilon,
               params.print_info,
               params.backend,
               params.input_filename,
               params.model_filename,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}
template std::ostream &operator<<(std::ostream &, const parameter<float> &);
template std::ostream &operator<<(std::ostream &, const parameter<double> &);

}  // namespace plssvm