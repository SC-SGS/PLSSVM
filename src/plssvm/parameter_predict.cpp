/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright
 */

#include "plssvm/parameter_predict.hpp"

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
parameter_predict<T>::parameter_predict(std::string input_filename, std::string model_filename) {
    base_type::input_filename = std::move(input_filename);
    base_type::model_filename = std::move(model_filename);
    predict_name_from_input();
}

template <typename T>
parameter_predict<T>::parameter_predict(int argc, char **argv) {
    // small helper function to convert a string to a lowercase string
    auto as_lowercase = [](const std::string_view str) -> std::string {
        std::string lowercase_str{ str };
        std::transform(str.begin(), str.end(), lowercase_str.begin(), [](const char c) { return std::tolower(c); });
        return lowercase_str;
    };

    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("test_file model_file [output_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("b,backend", "choose the backend: openmp|cuda|opencl|sycl", cxxopts::value<decltype(backend)>()->default_value(as_lowercase(fmt::format("{}", backend))))
            ("p,target_platform", "choose the target platform: automatic|cpu|gpu_nvidia|gpu_amd|gpu_intel", cxxopts::value<decltype(target)>()->default_value(as_lowercase(fmt::format("{}", target))))
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info)->default_value(fmt::format("{}", !print_info)))
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("test", "", cxxopts::value<decltype(input_filename)>(), "test_file")
            ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file")
            ("output", "", cxxopts::value<decltype(predict_filename)>(), "output_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "model", "output" });
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

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

    // parse print info
    print_info = !print_info;

    // parse test data filename
    if (!result.count("test")) {
        fmt::print(stderr, "Error missing test file!");
        fmt::print("{}", options.help());
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["test"].as<decltype(input_filename)>();

    // parse model filename
    if (!result.count("model")) {
        fmt::print(stderr, "Error missing model file!");
        fmt::print("{}", options.help());
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["model"].as<decltype(model_filename)>();

    // parse output filename
    if (result.count("output")) {
        model_filename = result["output"].as<decltype(predict_filename)>();
    } else {
        predict_name_from_input();
    }

    // TODO: parse other parameters
}

template <typename T>
void parameter_predict<T>::predict_name_from_input() {
    std::size_t pos = input_filename.find_last_of("/\\");
    predict_filename = input_filename.substr(pos + 1) + ".predict";
}

// explicitly instantiate template class
template class parameter_predict<float>;
template class parameter_predict<double>;

}  // namespace plssvm