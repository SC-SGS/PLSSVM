/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parser_predict.hpp"

#include "plssvm/backend_types.hpp"                                // plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_type.hpp"            // plssvm::sycl::list_available_sycl_implementations
#include "plssvm/constants.hpp"                                    // plssvm::real_type
#include "plssvm/detail/assert.hpp"                                // PLSSVM_ASSERT
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log
#include "plssvm/target_platforms.hpp"                             // plssvm::list_available_target_platforms
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity, plssvm::verbosity_level
#include "plssvm/version/version.hpp"                              // plssvm::version::detail::get_version_info

#include "cxxopts.hpp"    // cxxopts::{Options, value, ParseResult}
#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // fmt::format, fmt::join
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdlib>      // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cout, std::cerr, std::endl
#include <type_traits>  // std::is_same_v

namespace plssvm::detail::cmd {

parser_predict::parser_predict(int argc, char **argv) {
    // check for basic argc and argv correctness
    PLSSVM_ASSERT(argc >= 1, fmt::format("At least one argument is always given (the executable name), but argc is {}!", argc));
    PLSSVM_ASSERT(argv != nullptr, "At least one argument is always given (the executable name), but argv is a nullptr!");

    // setup command line parser with all available options
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("test_file model_file [output_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<backend_type>()->default_value(fmt::format("{}", backend)))
            ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<target_platform>()->default_value(fmt::format("{}", target)))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<sycl::implementation_type>()->default_value(fmt::format("{}", sycl_implementation_type)))
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
           ("performance_tracking", "the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr", cxxopts::value<decltype(performance_tracking_filename)>())
#endif
            ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
            ("verbosity", fmt::format("choose the level of verbosity: full|timing|libsvm|quiet (default: {})", fmt::format("{}", verbosity)), cxxopts::value<verbosity_level>())
            ("q,quiet", "quiet mode (no outputs regardless the provided verbosity level!)", cxxopts::value<bool>())
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
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: {}\n", e.what()) << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // print help message and exit
    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    // print version info
    if (result.count("version")) {
        std::cout << version::detail::get_version_info("plssvm-predict") << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: only up to three positional options may be given, but {} (\"{}\") additional option(s) where provided!", result.unmatched().size(), fmt::join(result.unmatched(), " ")) << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // parse SYCL implementation used in the SYCL backend
    sycl_implementation_type = result["sycl_implementation_type"].as<decltype(sycl_implementation_type)>();

    // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
    if (backend != backend_type::sycl && sycl_implementation_type != sycl::implementation_type::automatic) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}\n",
                    sycl_implementation_type);
    }
#endif

    // parse whether strings should be used as labels
    strings_as_labels = result["use_strings_as_labels"].as<decltype(strings_as_labels)>();

    // parse whether output is quiet or not
    const bool quiet = result["quiet"].as<bool>();

    // -q/--quiet has precedence over --verbosity
    if (result["verbosity"].count()) {
        const verbosity_level verb = result["verbosity"].as<verbosity_level>();
        if (quiet && verb != verbosity_level::quiet) {
            detail::log(verbosity_level::full | verbosity_level::warning,
                        "WARNING: explicitly set the -q/--quiet flag, but the provided verbosity level isn't \"quiet\"; setting --verbosity={} to --verbosity=quiet\n",
                        verb);
            verbosity = verbosity_level::quiet;
        } else {
            verbosity = verb;
        }
    } else if (quiet) {
        verbosity = verbosity_level::quiet;
    }

    // parse test data filename
    if (!result.count("test")) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing test file!\n") << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["test"].as<decltype(input_filename)>();

    // parse model filename
    if (!result.count("model")) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing model file!\n") << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    model_filename = result["model"].as<decltype(model_filename)>();

    // parse output filename
    if (result.count("output")) {
        predict_filename = result["output"].as<decltype(predict_filename)>();
    } else {
        const std::filesystem::path input_path{ input_filename };
        predict_filename = input_path.filename().string() + ".predict";
    }

    // parse performance tracking filename
    if (result.count("performance_tracking")) {
        performance_tracking_filename = result["performance_tracking"].as<decltype(performance_tracking_filename)>();
    }
}

std::ostream &operator<<(std::ostream &out, const parser_predict &params) {
    out << fmt::format(
        "backend: {}\n"
        "target platform: {}\n",
        params.backend,
        params.target);

    if (params.backend == backend_type::sycl || params.backend == backend_type::automatic) {
        out << fmt::format("SYCL implementation type: {}\n", params.sycl_implementation_type);
    }

    out << fmt::format(
        "label_type: {}\n"
        "real_type: {}\n"
        "input file (data set): '{}'\n"
        "input file (model): '{}'\n"
        "output file (prediction): '{}'\n",
        params.strings_as_labels ? "std::string" : "int (default)",
        std::is_same_v<real_type, float> ? "float" : "double (default)",
        params.input_filename,
        params.model_filename,
        params.predict_filename);

    if (!params.performance_tracking_filename.empty()) {
        out << fmt::format("performance tracking file: '{}'\n", params.performance_tracking_filename);
    }

    return out;
}

}  // namespace plssvm::detail::cmd
