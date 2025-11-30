/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parser_predict.hpp"

#include "plssvm/backend_types.hpp"                        // plssvm::list_available_backends
#include "plssvm/backends/Kokkos/execution_spaces.hpp"     // plssvm::kokkos::{list_available_execution_spaces, execution_space}
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::{list_available_sycl_data_parallel_kernels, data_parallel_kernels}
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::{list_available_sycl_implementations, implementation_type}
#include "plssvm/constants.hpp"                            // plssvm::real_type
#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/cmd/utility.hpp"                   // plssvm::detail::cmd::{filter_argv, kernel_type_help_message, parse_and_check_sycl_options_if_available,
                                                           // parse_and_check_kokkos_options_if_available, parse_and_check_mpi_options_if_available, parse_verbosity, max_cmd_width}
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::cmd_parser_exit
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform, plssvm::list_available_target_platforms
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity, plssvm::verbosity_level
#include "plssvm/version/version.hpp"                      // plssvm::version::detail::get_version_info

#include "cxxopts.hpp"   // cxxopts::Options, cxxopts::value, cxxopts::ParseResult
#include "fmt/color.h"   // fmt::fg, fmt::color::orange
#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <filesystem>   // std::filesystem::path
#include <iostream>     // std::cout, std::cerr, std::endl
#include <optional>     // std::optional
#include <type_traits>  // std::is_same_v
#include <utility>      // std::pair, std::move
#include <vector>       // std::vector

namespace plssvm::detail::cmd {

parser_predict::parser_predict(const mpi::communicator &comm, int argc, char **argv) {
    // check for basic argc and argv correctness
    PLSSVM_ASSERT(argc >= 1, fmt::format("At least one argument is always given (the executable name), but argc is {}!", argc));
    PLSSVM_ASSERT(argv != nullptr, "At least one argument is always given (the executable name), but argv is a nullptr!");

    // filter the command line arguments removing third party options
    std::vector<char *> filtered_args = filter_argv(argc, argv);

    // setup command line parser with all available options
    cxxopts::Options options("plssvm-predict", "LS-SVM with multiple (GPU-)backends");
    options
        .positional_help("test_file model_file [output_file]")
        .show_positional_help();
    options
        .set_width(max_cmd_width)
        .set_tab_expansion()
        // clang-format off
        .add_options()
            ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<backend_type>()->default_value(fmt::format("{}", backend)))
            ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<target_platform>()->default_value(fmt::format("{}", target)))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
            ("sycl_data_parallel_kernel", fmt::format("choose the data parallel kernel when using SYCL as backend: {}", fmt::join(sycl::list_available_sycl_data_parallel_kernels(), "|")), cxxopts::value<decltype(sycl_data_parallel_kernel)>()->default_value(fmt::format("{}", sycl_data_parallel_kernel)))
            ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<sycl::implementation_type>()->default_value(fmt::format("{}", sycl_implementation_type)))
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
            std::cout << version::detail::get_version_info("plssvm-predict") << std::endl;
        }
        throw cmd_parser_exit{ EXIT_SUCCESS };
    }

    // check if the number of positional arguments is not too large
    if (!result.unmatched().empty()) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: only up to three positional options may be given, but {} (\"{}\") additional option(s) where provided!", result.unmatched().size(), fmt::join(result.unmatched(), " ")) << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }

    // parse backend_type and cast the value to the respective enum
    backend = result["backend"].as<decltype(backend)>();

    // parse target_platform and cast the value to the respective enum
    target = result["target_platform"].as<decltype(target)>();

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

    // parse whether strings should be used as labels
    strings_as_labels = result["use_strings_as_labels"].as<decltype(strings_as_labels)>();

    // -q/--quiet has precedence over --verbosity
    const std::optional<verbosity_level> verb = parse_verbosity(result, comm);
    if (verb.has_value()) {
        verbosity = verb.value();
    }

    // parse test data filename
    if (!result.contains("test")) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing test file!\n") << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    input_filename = result["test"].as<decltype(input_filename)>();

    // parse model filename
    if (!result.contains("model")) {
        if (comm.is_main_rank()) {
            std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: missing model file!\n") << std::endl;
            std::cout << options.help() << std::endl;
        }
        throw cmd_parser_exit{ EXIT_FAILURE };
    }
    model_filename = result["model"].as<decltype(model_filename)>();

    // parse output filename
    if (result.contains("output")) {
        predict_filename = result["output"].as<decltype(predict_filename)>();
    } else {
        const std::filesystem::path input_path{ input_filename };
        predict_filename = input_path.filename().string() + ".predict";
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

std::ostream &operator<<(std::ostream &out, const parser_predict &params) {
    out << fmt::format(
        "backend: {}\n"
        "target platform: {}\n",
        params.backend,
        params.target);

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
    if (!params.mpi_load_balancing_weights.empty()) {
        out << fmt::format("mpi load-balancing weights: [{}]\n", fmt::join(params.mpi_load_balancing_weights, ", "));
    }

    return out;
}

}  // namespace plssvm::detail::cmd
