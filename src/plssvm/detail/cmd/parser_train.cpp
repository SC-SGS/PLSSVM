/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parser_train.hpp"

#include "plssvm/backend_types.hpp"                        // plssvm::list_available_backends
#include "plssvm/backends/Kokkos/execution_spaces.hpp"     // plssvm::kokkos::{list_available_execution_spaces, execution_space}
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::{list_available_sycl_data_parallel_kernels, data_parallel_kernels}
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::{list_available_sycl_implementations, implementation_type}
#include "plssvm/classification_types.hpp"                 // plssvm::classification_type, plssvm::classification_type_to_full_string
#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/cmd/utility.hpp"                   // plssvm::detail::cmd::{filter_argv, kernel_type_help_message, parse_and_check_sycl_options_if_available,
                                                           // parse_and_check_kokkos_options_if_available, parse_and_check_mpi_options_if_available, parse_verbosity, max_cmd_width}
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::cmd_parser_exit
#include "plssvm/gamma.hpp"                                // plssvm::get_gamma_string
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type, plssvm::kernel_function_type_to_math_string
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/svm_types.hpp"                            // plssvm::svm_type
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform, plssvm::list_available_target_platforms
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity, plssvm::verbosity_level
#include "plssvm/version/version.hpp"                      // plssvm::version::detail::get_version_info

#include "cxxopts.hpp"   // cxxopts::Options, cxxopts::value, cxxopts::ParseResult
#include "fmt/color.h"   // fmt::fg, fmt::color::red
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cout, std::cerr, std::endl
#include <optional>     // std::optional
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair, std::move
#include <variant>      // std::holds_alternative, std::get
#include <vector>       // std::vector

namespace plssvm::detail::cmd {

parser_train::parser_train(const mpi::communicator &comm, int argc, char **argv) {
    // check for basic argc and argv correctness
    PLSSVM_ASSERT(argc >= 1, fmt::format("At least one argument is always given (the executable name), but argc is {}!", argc));
    PLSSVM_ASSERT(argv != nullptr, "At least one argument is always given (the executable name), but argv is a nullptr!");

    // filter the command line arguments removing third party options
    std::vector<char *> filtered_args = filter_argv(argc, argv);

    // create the help message for the kernel function type
    const std::string kernel_type_help = kernel_type_help_message();

    cxxopts::Options options("plssvm-train", "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("training_set_file [model_file]")
        .show_positional_help();
    options
        .set_width(150)
        .set_tab_expansion()
        // clang-format off
       .add_options()
           ("s,svm_type", "set type of SVM\n\t 0 -- C-SVC\n\t 1 -- C-SVR", cxxopts::value<decltype(svm)>()->default_value(fmt::format("{}", detail::to_underlying(svm))))
           ("t,kernel_type", kernel_type_help, cxxopts::value<decltype(csvm_params.kernel_type)>()->default_value(fmt::format("{}", detail::to_underlying(csvm_params.kernel_type))))
           ("d,degree", "set degree in kernel function", cxxopts::value<decltype(csvm_params.degree)>()->default_value(fmt::format("{}", csvm_params.degree)))
           ("g,gamma", fmt::format("set gamma in kernel function (default: {})", get_gamma_string(csvm_params.gamma)), cxxopts::value<decltype(csvm_params.gamma)>())
           ("r,coef0", "set coef0 in kernel function", cxxopts::value<decltype(csvm_params.coef0)>()->default_value(fmt::format("{}", csvm_params.coef0)))
           ("c,cost", "set the parameter C", cxxopts::value<decltype(csvm_params.cost)>()->default_value(fmt::format("{}", csvm_params.cost)))
           ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<decltype(epsilon)>()->default_value(fmt::format("{}", epsilon)))
           ("i,max_iter", "set the maximum number of CG iterations (default: num_features)", cxxopts::value<long long int>())
           ("l,solver", "choose the solver: automatic|cg_explicit|cg_implicit", cxxopts::value<decltype(solver)>()->default_value("automatic"))
           ("a,classification", "the classification strategy to use for multi-class classification: oaa|oao", cxxopts::value<decltype(classification)>()->default_value(fmt::format("{}", classification)))
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<decltype(backend)>()->default_value(fmt::format("{}", backend)))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<decltype(target)>()->default_value(fmt::format("{}", target)))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_data_parallel_kernel", fmt::format("choose the data parallel kernel when using SYCL as backend: {}", fmt::join(sycl::list_available_sycl_data_parallel_kernels(), "|")), cxxopts::value<decltype(sycl_data_parallel_kernel)>()->default_value(fmt::format("{}", sycl_data_parallel_kernel)))
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<decltype(sycl_implementation_type)>()->default_value(fmt::format("{}", sycl_implementation_type)))
#endif
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
            ("kokkos_execution_space", fmt::format("choose the Kokkos execution space to be used in the Kokkos backend: {}", fmt::join(kokkos::list_available_execution_spaces(), "|")), cxxopts::value<decltype(kokkos_execution_space)>()->default_value(fmt::format("{}", kokkos_execution_space)))
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
           ("performance_tracking", "the output YAML file where the performance tracking results are written to; if not provided, the results are dumped to stderr", cxxopts::value<decltype(performance_tracking_filename)>())
#endif
#if defined(PLSSVM_HAS_MPI_ENABLED)
           ("mpi_load_balancing_weights", "can be used to load balance for MPI (must be integers); number of provided values must match the number of MPI ranks", cxxopts::value<decltype(mpi_load_balancing_weights)>())
#endif
           ("use_strings_as_labels", "use strings as labels for the classification task instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
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
        result = options.parse(static_cast<int>(filtered_args.size()), filtered_args.data());
    } catch (const std::exception &e) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: {}\n", e.what()) << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // print help message and exit
    if (result.contains("help")) {
        if (comm.is_main_rank()) {
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_SUCCESS };
    }

    // print version info
    if (result.contains("version")) {
        if (comm.is_main_rank()) {
            std::cout << version::detail::get_version_info("plssvm-train") << std::endl;
        }
        throw cmd_parser_exit{ EXIT_SUCCESS };
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: only up to two positional options may be given, but {} (\"{}\") additional option(s) where provided!\n", result.unmatched().size(), fmt::join(result.unmatched(), " ")) << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // parse svm_type and cast the value to the respective enum
    if (result.contains("svm_type")) {
        svm = result["svm_type"].as<decltype(svm)>();
    }

    // parse kernel_type and cast the value to the respective enum
    if (result.contains("kernel_type")) {
        csvm_params.kernel_type = result["kernel_type"].as<decltype(csvm_params.kernel_type)>();
    }

    // parse degree
    if (result.contains("degree")) {
        csvm_params.degree = result["degree"].as<decltype(csvm_params.degree)>();
    }

    // parse gamma
    if (result.contains("gamma")) {
        const decltype(csvm_params.gamma) gamma_input = result["gamma"].as<decltype(csvm_params.gamma)>();
        // check if the provided gamma is legal iff a real_type has been provided
        if (std::holds_alternative<real_type>(gamma_input) && std::get<real_type>(gamma_input) <= real_type{ 0.0 }) {
            if (comm.is_main_rank()) {
                std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: gamma must be greater than 0.0, but is {}!\n", std::get<real_type>(gamma_input)) << std::endl;
                std::cout << options.help() << std::endl;
            }
            throw cmd_parser_exit{ EXIT_FAILURE };
        }
        // provided gamma was legal -> override default value
        csvm_params.gamma = gamma_input;
    }

    // parse coef0
    if (result.contains("coef0")) {
        csvm_params.coef0 = result["coef0"].as<decltype(csvm_params.coef0)>();
    }

    // parse cost
    if (result.contains("cost")) {
        csvm_params.cost = result["cost"].as<decltype(csvm_params.cost)>();
    }

    // parse epsilon
    if (result.contains("epsilon")) {
        epsilon = result["epsilon"].as<decltype(epsilon)>();
    }

    // parse max_iter
    if (result.contains("max_iter")) {
        const auto max_iter_input = result["max_iter"].as<long long int>();
        // check if the provided max_iter is legal
        if (max_iter_input <= decltype(max_iter_input){ 0 }) {
            if (comm.is_main_rank()) {
                std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: max_iter must be greater than 0, but is {}!\n", max_iter_input) << std::endl;
                std::cout << options.help() << std::endl;
            }
            throw cmd_parser_exit{ EXIT_FAILURE };
        }
        // provided max_iter was legal -> override default value
        max_iter = static_cast<decltype(max_iter)>(max_iter_input);
    }

    // parse the classification type
    if (result.contains("classification")) {
        classification = result["classification"].as<decltype(classification)>();

        // warn if a classification type has been provided, but the SVM type is a C-SVR (regression)
        if (svm == svm_type::csvr) {
            detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                  comm,
                                  "WARNING: explicitly set a classification type but the current svm_type is a C-SVR; ignoring --classification={}\n",
                                  classification);
        }
    }

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

    // parse the solver_type and cast the value to the respective enum
    solver = result["solver"].as<decltype(solver)>();

    // parse the SYCL related options
    const std::optional<std::pair<sycl::data_parallel_kernel, sycl::implementation_type>> sycl_options = parse_and_check_sycl_options_if_available(result, comm, backend, target);
    if (sycl_options.has_value()) {
        sycl_data_parallel_kernel = sycl_options->first;
        sycl_implementation_type = sycl_options->second;
    }

    // parse the Kokkos related options
    const std::optional<kokkos::execution_space> kokkos_options = parse_and_check_kokkos_options_if_available(result, comm, backend, target);
    if (kokkos_options.has_value()) {
        kokkos_execution_space = kokkos_options.value();
    }

    // parse whether strings should be used as labels for the classification task
    strings_as_labels = result["use_strings_as_labels"].as<decltype(strings_as_labels)>();
    if (svm != svm_type::csvc && strings_as_labels) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              comm,
                              "WARNING: explicitly requested string labels for the regression task; ignoring --use_strings_as_labels\n");
    }

    // -q/--quiet has precedence over --verbosity
    const std::optional<verbosity_level> verb = parse_verbosity(result, comm);
    if (verb.has_value()) {
        verbosity = verb.value();
    }

    // parse input data filename
    if (!result.contains("input")) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing input file!\n") << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    input_filename = result["input"].as<decltype(input_filename)>();

    // parse output model filename
    if (result.contains("model")) {
        model_filename = result["model"].as<decltype(model_filename)>();
    } else {
        const std::filesystem::path input_path{ input_filename };
        model_filename = input_path.filename().string() + ".model";
    }

    // parse performance tracking filename
    if (result.contains("performance_tracking")) {
        performance_tracking_filename = result["performance_tracking"].as<decltype(performance_tracking_filename)>();
    }

    // parse the MPI related options
    std::optional<std::vector<std::size_t>> mpi_options = parse_and_check_mpi_options_if_available(result, options, comm);
    if (mpi_options.has_value()) {
        mpi_load_balancing_weights = std::move(mpi_options.value());
    }
}

std::ostream &operator<<(std::ostream &out, const parser_train &params) {
    out << fmt::format("svm_type: {}\n"
                       "kernel_type: {} -> {}\n",
                       params.svm,
                       params.csvm_params.kernel_type,
                       kernel_function_type_to_math_string(params.csvm_params.kernel_type));
    switch (params.csvm_params.kernel_type) {
        case kernel_function_type::linear:
            break;
        case kernel_function_type::polynomial:
            out << fmt::format("degree: {}\n"
                               "gamma: {}\n"
                               "coef0: {}\n",
                               params.csvm_params.degree,
                               get_gamma_string(params.csvm_params.gamma),
                               params.csvm_params.coef0);
            break;
        case kernel_function_type::rbf:
        case kernel_function_type::laplacian:
        case kernel_function_type::chi_squared:
            out << fmt::format("gamma: {}\n", get_gamma_string(params.csvm_params.gamma));
            break;
        case kernel_function_type::sigmoid:
            out << fmt::format("gamma: {}\n"
                               "coef0: {}\n",
                               get_gamma_string(params.csvm_params.gamma),
                               params.csvm_params.coef0);
            break;
    }
    out << fmt::format("cost: {}\n", params.csvm_params.cost);
    out << fmt::format("epsilon: {}\n", params.epsilon);
    if (params.max_iter == 0) {
        out << "max_iter: num_data_points\n";
    } else {
        out << fmt::format("max_iter: {}\n", params.max_iter);
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
            "SYCL data parallel kernel: {}\n",
            params.sycl_implementation_type,
            params.sycl_data_parallel_kernel);
    }

    if (params.backend == backend_type::kokkos || params.backend == backend_type::automatic) {
        out << fmt::format("Kokkos execution space: {}\n", params.kokkos_execution_space);
    }

    out << fmt::format(
        "classification_type: {}\n"
        "label_type: {}\n"
        "real_type: {}\n"
        "input file (data set): '{}'\n"
        "output file (model): '{}'\n",
        classification_type_to_full_string(params.classification),
        params.strings_as_labels ? "std::string" : "int",
        std::is_same_v<real_type, float> ? "float" : "double",
        params.input_filename,
        params.model_filename);

    if (!params.performance_tracking_filename.empty()) {
        out << fmt::format("performance tracking file: '{}'\n", params.performance_tracking_filename);
    }
    if (!params.mpi_load_balancing_weights.empty()) {
        out << fmt::format("mpi load-balancing weights: [{}]\n", fmt::join(params.mpi_load_balancing_weights, ", "));
    }

    return out;
}

}  // namespace plssvm::detail::cmd
