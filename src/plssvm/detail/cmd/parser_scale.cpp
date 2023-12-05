/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parser_scale.hpp"

#include "plssvm/detail/assert.hpp"                                // PLSSVM_ASSERT
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity, plssvm::verbosity_level
#include "plssvm/version/version.hpp"                              // plssvm::version::detail::get_version_info

#include "cxxopts.hpp"    // cxxopts::{Options, value, ParseResult}
#include "fmt/core.h"     // fmt::format, fmt::join
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdlib>      // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cout, std::cerr, std::endl
#include <type_traits>  // std::is_same_v

namespace plssvm::detail::cmd {

parser_scale::parser_scale(int argc, char **argv) {
    // check for basic argc and argv correctness
    PLSSVM_ASSERT(argc >= 1, fmt::format("At least one argument is always given (the executable name), but argc is {}!", argc));
    PLSSVM_ASSERT(argv != nullptr, "At least one argument is always given (the executable name), but argv is a nullptr!");

    // setup command line parser with all available options
    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("input_file [scaled_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
       .add_options()
           ("l,lower", "lower is the lowest (minimal) value allowed in each dimension", cxxopts::value<decltype(lower)>()->default_value(fmt::format("{}", lower)))
           ("u,upper", "upper is the highest (maximal) value allowed in each dimension", cxxopts::value<decltype(upper)>()->default_value(fmt::format("{}", upper)))
           ("f,format", "the file format to output the scaled data set to", cxxopts::value<decltype(format)>()->default_value(fmt::format("{}", format)))
           ("s,save_filename", "the file to which the scaling factors should be saved", cxxopts::value<decltype(save_filename)>())
           ("r,restore_filename", "the file from which previous scaling factors should be loaded", cxxopts::value<decltype(restore_filename)>())
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
           ("performance_tracking", "the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr", cxxopts::value<decltype(performance_tracking_filename)>())
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
           ("verbosity", fmt::format("choose the level of verbosity: full|timing|libsvm|quiet (default: {})", fmt::format("{}", verbosity)), cxxopts::value<verbosity_level>())
           ("q,quiet", "quiet mode (no outputs regardless the provided verbosity level!)", cxxopts::value<bool>())
           ("h,help", "print this helper message", cxxopts::value<bool>())
           ("v,version", "print version information", cxxopts::value<bool>())
           ("input", "", cxxopts::value<decltype(input_filename)>(), "input_file")
           ("scaled", "", cxxopts::value<decltype(scaled_filename)>(), "scaled_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "scaled" });
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
        std::cout << version::detail::get_version_info("plssvm-scale", false) << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: only up to two positional options may be given, but {} (\"{}\") additional option(s) where provided!\n", result.unmatched().size(), fmt::join(result.unmatched(), " ")) << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse the lowest allowed value
    lower = result["lower"].as<decltype(lower)>();

    // parse the highest allowed value
    upper = result["upper"].as<decltype(upper)>();

    // lower must be strictly less than upper!
    if (lower >= upper) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: invalid scaling range [lower, upper] with [{}, {}]!\n", lower, upper) << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse the file format
    format = result["format"].as<decltype(format)>();

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

    // parse input data filename
    if (!result.count("input")) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing input file!\n") << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.count("scaled")) {
        scaled_filename = result["scaled"].as<decltype(scaled_filename)>();
    }

    // can only use one of save_filename or restore_filename
    if (result.count("save_filename") && result.count("restore_filename")) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: cannot use -s (--save_filename) and -r (--restore_filename) simultaneously!\n") << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse the file name to save the calculated weights to
    if (result.count("save_filename")) {
        save_filename = result["save_filename"].as<decltype(save_filename)>();
    }

    // parse the file name to restore the previously saved weights from
    if (result.count("restore_filename")) {
        if (result.count("lower") || result.count("upper")) {
            detail::log(verbosity_level::full | verbosity_level::warning,
                        "WARNING: provided -l (--lower) and/or -u (--upper) together with -r (--restore_filename); ignoring -l/-u\n");
        }
        restore_filename = result["restore_filename"].as<decltype(restore_filename)>();
    }

    // parse performance tracking filename
    if (result.count("performance_tracking")) {
        performance_tracking_filename = result["performance_tracking"].as<decltype(performance_tracking_filename)>();
    }
}

std::ostream &operator<<(std::ostream &out, const parser_scale &params) {
    out << fmt::format(
        "lower: {}\n"
        "upper: {}\n"
        "output file format: {}\n"
        "label_type: {}\n"
        "real_type: {}\n"
        "input file: '{}'\n"
        "scaled file: '{}'\n"
        "save file (scaling factors): '{}'\n"
        "restore file (scaling factors): '{}'\n",
        params.lower,
        params.upper,
        params.format,
        params.strings_as_labels ? "std::string" : "int (default)",
        std::is_same_v<real_type, float> ? "float" : "double (default)",
        params.input_filename,
        params.scaled_filename,
        params.save_filename,
        params.restore_filename);
    if (!params.performance_tracking_filename.empty()) {
        out << fmt::format("performance tracking file: '{}'\n", params.performance_tracking_filename);
    }
    return out;
}

}  // namespace plssvm::detail::cmd