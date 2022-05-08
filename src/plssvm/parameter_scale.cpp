/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter_scale.hpp"

#include "plssvm/backend_types.hpp"                      // plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl_generic::list_available_sycl_implementations
#include "plssvm/constants.hpp"                          // plssvm::verbose
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::to_underlying
#include "plssvm/target_platforms.hpp"                   // plssvm::list_available_target_platforms

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>      // stderr
#include <cstdlib>     // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>   // std::exception
#include <filesystem>  // std::filesystem::path

namespace plssvm {

template <typename T>
parameter_scale<T>::parameter_scale(int argc, char **argv) {
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("input_file scaled_file")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("l,lower", "lower is the lowest (minimal) value allowed in each dimension", cxxopts::value<decltype(lower)>()->default_value(fmt::format("{}", lower)))
            ("u,upper", "upper is the highest (maximal) value allowed in each dimension", cxxopts::value<decltype(upper)>()->default_value(fmt::format("{}", upper)))
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(plssvm::verbose)->default_value(fmt::format("{}", !plssvm::verbose)))
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("input", "", cxxopts::value<decltype(input_filename)>(), "input_file")
            ("scaled", "", cxxopts::value<decltype(scaled_filename)>(), "scaled_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "scaled" });
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

    // parse the lowest allowed value
    lower = result["lower"].as<decltype(lower)>();

    // parse the highest allowed value
    upper = result["upper"].as<decltype(upper)>();

    // parse whether output is quiet or not
    plssvm::verbose = !plssvm::verbose;

    // parse input data filename
    if (!result.count("input")) {
        fmt::print(stderr, "Error missing input file!");
        fmt::print("{}", options.help());
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.count("scaled")) {
        scaled_filename = result["scaled"].as<decltype(scaled_filename)>();
    } else {
        scaled_filename = this->scaled_name_from_input();
    }
}

template <typename T>
[[nodiscard]] std::string parameter_scale<T>::scaled_name_from_input() {
    auto input_path = std::filesystem::path(input_filename);
    return input_path.replace_filename(input_path.filename().string() + ".scaled").string();
}

// explicitly instantiate template class
template class parameter_scale<float>;
template class parameter_scale<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_scale<T> &params) {
    out << static_cast<const parameter<T>&>(params);
    return out << fmt::format(
               "input_filename              '{}'\n"
               "scale_filename              '{}'\n",
               params.input_filename,
               params.scaled_filename);
}
template std::ostream &operator<<(std::ostream &, const parameter_scale<float> &);
template std::ostream &operator<<(std::ostream &, const parameter_scale<double> &);

}  // namespace plssvm