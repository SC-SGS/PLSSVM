/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter_predict.hpp"

#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl_generic::list_available_sycl_implementations
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::as_lower_case
#include "plssvm/parameter.hpp"                          // plssvm::parameter
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata{is_populated, commit_sha1, commit_date, branch}
#include "plssvm/version/version.hpp"                    // plssvm::version::{name, version, target_platforms}

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>     // stderr
#include <cstdlib>    // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <string>     // std::string
#include <utility>    // std::move

namespace plssvm {

template <typename T>
parameter_predict<T>::parameter_predict(std::string p_input_filename, std::string p_model_filename) {
    base_type::input_filename = std::move(p_input_filename);
    base_type::model_filename = std::move(p_model_filename);
    base_type::predict_filename = base_type::predict_name_from_input();

    base_type::parse_model_file(base_type::model_filename);
    base_type::parse_test_file(base_type::input_filename);
}

template <typename T>
parameter_predict<T>::parameter_predict(int argc, char **argv) {
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("test_file model_file [output_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<decltype(backend)>()->default_value(detail::as_lower_case(fmt::format("{}", backend))))
            ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<decltype(target)>()->default_value(detail::as_lower_case(fmt::format("{}", target))))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<decltype(sycl_implementation_type)>()->default_value(detail::as_lower_case(fmt::format("{}", sycl_implementation_type))))
#endif
            ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(print_info)->default_value(fmt::format("{}", !print_info)))
            ("h,help", "print this helper message", cxxopts::value<bool>())
            ("v,version", "print version information", cxxopts::value<bool>())
            ("test", "", cxxopts::value<decltype(input_filename)>(), "test_file")
            ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file")
            ("output", "", cxxopts::value<decltype(predict_filename)>(), "output_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "test", "model", "output" });
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

    // print version info
    if (result.count("version")) {
        fmt::print("plssvm-predict v{} ", version::version);
        if (version::git_metadata::is_populated()) {
            std::string_view date = version::git_metadata::commit_date();
            date.remove_suffix(date.size() - date.find_last_of(' '));
            fmt::print("({} {} {} ({}))", version::git_metadata::remote_url(), version::git_metadata::branch(), version::git_metadata::commit_sha1(), date);
        }
        fmt::print("\n\n{}\n", version::name);
        fmt::print("  PLSSVM_TARGET_PLATFORMS: {}\n", version::detail::target_platforms);
        fmt::print("  available backends: {}\n", fmt::join(list_available_backends(), ", "));
#if defined(PLSSVM_HAS_SYCL_BACKEND)
        fmt::print("  available SYCL implementations: {}\n", sycl::list_available_sycl_implementations());
#endif
        fmt::print("\nCopyright(C) 2018-today The PLSSVM project - All Rights Reserved\nThis is free software distributed under the MIT license; see the source for more information.\n");
        std::exit(EXIT_SUCCESS);
    }

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // parse SYCL implementation used in the SYCL backend
    sycl_implementation_type = result["sycl_implementation_type"].as<decltype(sycl_implementation_type)>();
#endif

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
    model_filename = result["model"].as<decltype(model_filename)>();

    // parse output filename
    if (result.count("output")) {
        predict_filename = result["output"].as<decltype(predict_filename)>();
    } else {
        predict_filename = base_type::predict_name_from_input();
    }

    base_type::parse_model_file(model_filename);
    base_type::parse_test_file(input_filename);
}

// explicitly instantiate template class
template class parameter_predict<float>;
template class parameter_predict<double>;

}  // namespace plssvm
