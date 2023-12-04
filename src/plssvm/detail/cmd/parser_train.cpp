/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parser_train.hpp"

#include "plssvm/backend_types.hpp"                                // plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_type.hpp"            // plssvm::sycl_generic::list_available_sycl_implementations
#include "plssvm/classification_types.hpp"                         // plssvm::classification_type, plssvm::classification_type_to_full_string
#include "plssvm/constants.hpp"                                    // plssvm::real_type
#include "plssvm/default_value.hpp"                                // plssvm::default_value
#include "plssvm/detail/assert.hpp"                                // PLSSVM_ASSERT
#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log
#include "plssvm/detail/string_utility.hpp"                        // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"                               // plssvm::detail::to_underlying
#include "plssvm/kernel_function_types.hpp"                        // plssvm::kernel_type_to_math_string
#include "plssvm/solver_types.hpp"                                 // plssvm::solver_types
#include "plssvm/target_platforms.hpp"                             // plssvm::list_available_target_platforms
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity, plssvm::verbosity_level
#include "plssvm/version/version.hpp"                              // plssvm::version::detail::get_version_info

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/color.h"    // fmt::fg, fmt::color::red
#include "fmt/core.h"     // fmt::format, fmt::join
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdlib>      // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cout, std::cerr, std::endl
#include <type_traits>  // std::is_same_v

namespace plssvm::detail::cmd {

parser_train::parser_train(int argc, char **argv) {
    // check for basic argc and argv correctness
    PLSSVM_ASSERT(argc >= 1, fmt::format("At least one argument is always given (the executable name), but argc is {}!", argc));
    PLSSVM_ASSERT(argv != nullptr, "At least one argument is always given (the executable name), but argv is a nullptr!");

    cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("training_set_file [model_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
       .add_options()
           ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear: u'*v\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<typename decltype(csvm_params.kernel_type)::value_type>()->default_value(fmt::format("{}", detail::to_underlying(csvm_params.kernel_type))))
           ("d,degree", "set degree in kernel function", cxxopts::value<typename decltype(csvm_params.degree)::value_type>()->default_value(fmt::format("{}", csvm_params.degree)))
           ("g,gamma", "set gamma in kernel function (default: 1 / num_features)", cxxopts::value<typename decltype(csvm_params.gamma)::value_type>())
           ("r,coef0", "set coef0 in kernel function", cxxopts::value<typename decltype(csvm_params.coef0)::value_type>()->default_value(fmt::format("{}", csvm_params.coef0)))
           ("c,cost", "set the parameter C", cxxopts::value<typename decltype(csvm_params.cost)::value_type>()->default_value(fmt::format("{}", csvm_params.cost)))
           ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<typename decltype(epsilon)::value_type>()->default_value(fmt::format("{}", epsilon)))
           ("i,max_iter", "set the maximum number of CG iterations (default: num_features)", cxxopts::value<long long int>())
           ("l,solver", "choose the solver: automatic|cg_explicit|cg_streaming|cg_implicit", cxxopts::value<decltype(solver)>()->default_value("automatic"))
           ("a,classification", "the classification strategy to use for multi-class classification: oaa|oao", cxxopts::value<typename decltype(classification)::value_type>()->default_value(fmt::format("{}", classification)))
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<decltype(backend)>()->default_value(fmt::format("{}", backend)))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<decltype(target)>()->default_value(fmt::format("{}", target)))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_kernel_invocation_type", "choose the kernel invocation type when using SYCL as backend: automatic|nd_range", cxxopts::value<decltype(sycl_kernel_invocation_type)>()->default_value(fmt::format("{}", sycl_kernel_invocation_type)))
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<decltype(sycl_implementation_type)>()->default_value(fmt::format("{}", sycl_implementation_type)))
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
           ("performance_tracking", "the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr", cxxopts::value<decltype(performance_tracking_filename)>())
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
           ("verbosity", fmt::format("choose the level of verbosity: full|timing|libsvm|quiet (default: {})", fmt::format("{}", verbosity)), cxxopts::value<verbosity_level>())
           ("q,quiet", "quiet mode (no outputs regardless the provided verbosity level!)", cxxopts::value<bool>())
           ("h,help", "print this helper message", cxxopts::value<bool>())
           ("v,version", "print version information", cxxopts::value<bool>())
           ("input", "", cxxopts::value<decltype(input_filename)>(), "training_set_file")
           ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file");
    // clang-format on

    // parse command line options
    cxxopts::ParseResult result;
    try {
        options.parse_positional({ "input", "model" });
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
        std::cout << version::detail::get_version_info("plssvm-train") << std::endl;
        std::exit(EXIT_SUCCESS);
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: only up to two positional options may be given, but {} (\"{}\") additional option(s) where provided!\n", result.unmatched().size(), fmt::join(result.unmatched(), " ")) << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // parse kernel_type and cast the value to the respective enum
    if (result.count("kernel_type")) {
        csvm_params.kernel_type = result["kernel_type"].as<typename decltype(csvm_params.kernel_type)::value_type>();
    }

    // parse degree
    if (result.count("degree")) {
        csvm_params.degree = result["degree"].as<typename decltype(csvm_params.degree)::value_type>();
    }

    // parse gamma
    if (result.count("gamma")) {
        const typename decltype(csvm_params.gamma)::value_type gamma_input = result["gamma"].as<typename decltype(csvm_params.gamma)::value_type>();
        // check if the provided gamma is legal
        if (gamma_input <= decltype(gamma_input){ 0.0 }) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: gamma must be greater than 0.0, but is {}!\n", gamma_input) << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // provided gamma was legal -> override default value
        csvm_params.gamma = gamma_input;
    }

    // parse coef0
    if (result.count("coef0")) {
        csvm_params.coef0 = result["coef0"].as<typename decltype(csvm_params.coef0)::value_type>();
    }

    // parse cost
    if (result.count("cost")) {
        csvm_params.cost = result["cost"].as<typename decltype(csvm_params.cost)::value_type>();
    }

    // parse epsilon
    if (result.count("epsilon")) {
        epsilon = result["epsilon"].as<typename decltype(epsilon)::value_type>();
    }

    // parse max_iter
    if (result.count("max_iter")) {
        const auto max_iter_input = result["max_iter"].as<long long int>();
        // check if the provided max_iter is legal
        if (max_iter_input <= decltype(max_iter_input){ 0 }) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: max_iter must be greater than 0, but is {}!\n", max_iter_input) << std::endl;
            std::cout << options.help() << std::endl;
            std::exit(EXIT_FAILURE);
        }
        // provided max_iter was legal -> override default value
        max_iter = static_cast<typename decltype(max_iter)::value_type>(max_iter_input);
    }

    // parse the classification type
    if (result.count("classification")) {
        classification = result["classification"].as<typename decltype(classification)::value_type>();
    }

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

    // parse the solver_type and cast the value to the respective enum
    solver = result["solver"].as<decltype(solver)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // parse kernel invocation type when using SYCL as backend
    sycl_kernel_invocation_type = result["sycl_kernel_invocation_type"].as<decltype(sycl_kernel_invocation_type)>();

    // warn if kernel invocation type is explicitly set but SYCL isn't the current backend
    if (backend != backend_type::sycl && sycl_kernel_invocation_type != sycl::kernel_invocation_type::automatic) {
        detail::log(verbosity_level::full | verbosity_level::warning,
                    "WARNING: explicitly set a SYCL kernel invocation type but the current backend isn't SYCL; ignoring --sycl_kernel_invocation_type={}\n",
                    sycl_kernel_invocation_type);
    }

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

    // parse input data filename
    if (!result.count("input")) {
        std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing input file!\n") << std::endl;
        std::cout << options.help() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.count("model")) {
        model_filename = result["model"].as<decltype(model_filename)>();
    } else {
        const std::filesystem::path input_path{ input_filename };
        model_filename = input_path.filename().string() + ".model";
    }

    // parse performance tracking filename
    if (result.count("performance_tracking")) {
        performance_tracking_filename = result["performance_tracking"].as<decltype(performance_tracking_filename)>();
    }
}

std::ostream &operator<<(std::ostream &out, const parser_train &params) {
    out << fmt::format("kernel_type: {} -> {}\n", params.csvm_params.kernel_type, kernel_function_type_to_math_string(params.csvm_params.kernel_type));
    switch (params.csvm_params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial: {
            if (params.csvm_params.gamma.is_default()) {
                out << "gamma: 1 / num_features (default)\n";
            } else {
                out << fmt::format("gamma: {}\n", params.csvm_params.gamma.value());
            }
            out << fmt::format("coef0: {}{}\n", params.csvm_params.coef0.value(), params.csvm_params.coef0.is_default() ? " (default)" : "");
            out << fmt::format("degree: {}{}\n", params.csvm_params.degree.value(), params.csvm_params.degree.is_default() ? " (default)" : "");
        } break;
        case kernel_function_type::rbf:
            if (params.csvm_params.gamma.is_default()) {
                out << "gamma: 1 / num_features (default)\n";
            } else {
                out << fmt::format("gamma: {}\n", params.csvm_params.gamma.value());
            }
            break;
    }
    out << fmt::format("cost: {}{}\n", params.csvm_params.cost.value(), params.csvm_params.cost.is_default() ? " (default)" : "");
    out << fmt::format("epsilon: {}{}\n", params.epsilon.value(), params.epsilon.is_default() ? " (default)" : "");
    if (params.max_iter.is_default()) {
        out << "max_iter: num_data_points (default)\n";
    } else {
        out << fmt::format("max_iter: {}\n", params.max_iter.value());
    }

    out << fmt::format(
        "backend: {}\n"
        "target platform: {}\n"
        "solver: {}\n",
        params.backend,
        params.target,
        params.solver);

    if (params.backend == backend_type::sycl || params.backend == backend_type::automatic) {
        out << fmt::format(
            "SYCL implementation type: {}\n"
            "SYCL kernel invocation type: {}\n",
            params.sycl_implementation_type,
            params.sycl_kernel_invocation_type);
    }

    out << fmt::format(
        "classification_type: {}{}\n"
        "label_type: {}\n"
        "real_type: {}\n"
        "input file (data set): '{}'\n"
        "output file (model): '{}'\n",
        classification_type_to_full_string(params.classification.value()),
        params.classification.is_default() ? " (default)" : "",
        params.strings_as_labels ? "std::string" : "int (default)",
        std::is_same_v<real_type, float> ? "float" : "double (default)",
        params.input_filename,
        params.model_filename);
    if (!params.performance_tracking_filename.empty()) {
        out << fmt::format("performance tracking file: '{}'\n", params.performance_tracking_filename);
    }
    return out;
}

}  // namespace plssvm::detail::cmd