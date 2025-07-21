/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/utility.hpp"

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type, plssvm::list_available_backends, plssvm::determine_default_backend
#include "plssvm/backends/Kokkos/execution_space.hpp"      // plssvm::kokkos::execution_space
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernels
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::implementation_type
#include "plssvm/detail/logging/mpi_log_untracked.hpp"     // plssvm::detail::log_untracked
#include "plssvm/detail/string_utility.hpp"                // plssvm::detail::starts_with
#include "plssvm/detail/utility.hpp"                       // plssvm::detail::to_underlying
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::cmd_parser_exit
#include "plssvm/kernel_function_types.hpp"                // plssvm::kernel_function_type
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform, plssvm::determine_default_target_platform
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "cxxopts.hpp"   // cxxopts::ParseResult, cxxopts::Options
#include "fmt/color.h"   // fmt::fg, fmt::color::red
#include "fmt/format.h"  // fmt::format

#include <cstddef>   // std::size_t
#include <cstdlib>   // EXIT_FAILURE
#include <iostream>  // std::cout, std::cerr, std::endl
#include <optional>  // std::optional, std::nullopt
#include <string>    // std::string
#include <utility>   // std::pair, std::make_pair
#include <vector>    // std::vector

namespace plssvm::detail::cmd {

std::vector<char *> filter_argv(int argc, char **argv, const std::vector<std::string> &prefix_filter) {
    // We ignore all command line options starting with --hpx: like --hpx:threads=42.
    // We also ignore all command line options starting with --kokkos-.
    // NOTE: this does not include OUR command line option --kokkos_execution_space.
    std::vector<char *> filtered_argv{ argv[0] };
    for (std::size_t i = 1; i < static_cast<std::size_t>(argc); ++i) {
        bool remove_option = false;

        // check whether the current command line option starts with any of the provided prefixes
        for (const std::string &prefix : prefix_filter) {
            if (detail::starts_with(argv[i], prefix)) {
                remove_option = true;
            }
        }

        // only add the command line options that should not be removed
        if (!remove_option) {
            filtered_argv.push_back(argv[i]);
        }
    }

    return filtered_argv;
}

std::string kernel_type_help_message() {
    // create the help message for the kernel function type
    const auto kernel_type_to_help_entry = [](const kernel_function_type kernel) {
        return fmt::format("\t {} -- {}: {}\n", detail::to_underlying(kernel), kernel, kernel_function_type_to_math_string(kernel));
    };
    std::string kernel_type_help{ "set type of kernel function. \n" };
    for (const kernel_function_type kernel : { kernel_function_type::linear, kernel_function_type::polynomial, kernel_function_type::rbf, kernel_function_type::sigmoid, kernel_function_type::laplacian, kernel_function_type::chi_squared }) {
        kernel_type_help += kernel_type_to_help_entry(kernel);
    }
    kernel_type_help.pop_back();  // remove last newline character

    return kernel_type_help;
}

std::optional<std::pair<sycl::data_parallel_kernel, sycl::implementation_type>> parse_and_check_sycl_options_if_available([[maybe_unused]] const cxxopts::ParseResult &result, [[maybe_unused]] const mpi::communicator &comm, [[maybe_unused]] const backend_type backend, [[maybe_unused]] const target_platform target) {
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    // parse the data parallel kernel when using SYCL as backend
    const sycl::data_parallel_kernel data_parallel_kernel = result["sycl_data_parallel_kernel"].as<sycl::data_parallel_kernel>();

    // assemble warning condition
    const std::vector<target_platform> target_platforms = { target == target_platform::automatic ? determine_default_target_platform() : target };
    const bool sycl_backend_is_used = backend == backend_type::sycl || (backend == backend_type::automatic && determine_default_backend(list_available_backends(), target_platforms) == backend_type::sycl);

    // warn if the data parallel kernel is explicitly set but SYCL isn't the current (automatic) backend
    if (!sycl_backend_is_used && data_parallel_kernel != sycl::data_parallel_kernel::automatic) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              comm,
                              "WARNING: explicitly set a SYCL data parallel kernel but the current backend isn't SYCL; ignoring --sycl_data_parallel_kernel={}\n",
                              data_parallel_kernel);
    }

    // parse the SYCL implementation used in the SYCL backend
    const sycl::implementation_type implementation_type = result["sycl_implementation_type"].as<sycl::implementation_type>();

    // warn if a SYCL implementation type is explicitly set but SYCL isn't the current (automatic) backend
    if (!sycl_backend_is_used && implementation_type != sycl::implementation_type::automatic) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              comm,
                              "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}\n",
                              implementation_type);
    }

    return std::make_pair(data_parallel_kernel, implementation_type);
#else
    return std::nullopt;
#endif
}

[[nodiscard]] std::optional<kokkos::execution_space> parse_and_check_kokkos_options_if_available([[maybe_unused]] const cxxopts::ParseResult &result, [[maybe_unused]] const mpi::communicator &comm, [[maybe_unused]] const backend_type backend, [[maybe_unused]] const target_platform target) {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    // parse execution space when using Kokkos as backend
    const kokkos::execution_space execution_space = result["kokkos_execution_space"].as<kokkos::execution_space>();

    // assemble warning condition
    const std::vector<target_platform> target_platforms = { target == target_platform::automatic ? determine_default_target_platform() : target };
    const bool kokkos_backend_is_used = backend == backend_type::kokkos || (backend == backend_type::automatic && determine_default_backend(list_available_backends(), target_platforms) == backend_type::kokkos);

    // warn if the kokkos execution space is explicitly set but Kokkos isn't the current (automatic) backend
    if (!kokkos_backend_is_used && execution_space != kokkos::execution_space::automatic) {
        detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                              comm,
                              "WARNING: explicitly set a Kokkos execution space but the current backend isn't Kokkos; ignoring --kokkos_execution_space={}\n",
                              execution_space);
    }

    return execution_space;
#else
    return std::nullopt;
#endif
}

std::optional<std::vector<std::size_t>> parse_and_check_mpi_options_if_available([[maybe_unused]] const cxxopts::ParseResult &result, [[maybe_unused]] const cxxopts::Options &options, [[maybe_unused]] const mpi::communicator &comm) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    // parse MPI load balancing factors
    if (result.count("mpi_load_balancing_weights")) {
        std::vector<std::size_t> mpi_load_balancing_weights = result["mpi_load_balancing_weights"].as<std::vector<std::size_t>>();

        // sanity-check provided balance factors
        if (mpi_load_balancing_weights.size() != comm.size()) {
            if (comm.is_main_rank()) {
                std::cerr << fmt::format(fmt::fg(fmt::color::red), "ERROR: the number of load balancing weights ({}) must match the number of MPI ranks ({})!\n", mpi_load_balancing_weights.size(), comm.size()) << std::endl;
                std::cout << options.help() << std::endl;
            }
            throw cmd_parser_exit{ EXIT_FAILURE };
        }
        return std::make_optional(std::move(mpi_load_balancing_weights));
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

std::optional<verbosity_level> parse_verbosity(const cxxopts::ParseResult &result, const mpi::communicator &comm) {
    // parse whether output is quiet or not
    const bool quiet = result["quiet"].as<bool>();

    if (result["verbosity"].count()) {
        const verbosity_level verb = result["verbosity"].as<verbosity_level>();
        if (quiet && verb != verbosity_level::quiet) {
            detail::log_untracked(verbosity_level::full | verbosity_level::warning,
                                  comm,
                                  "WARNING: explicitly set the -q/--quiet flag, but the provided verbosity level isn't \"quiet\"; setting --verbosity={} to --verbosity=quiet\n",
                                  verb);
            return verbosity_level::quiet;
        } else {
            return verb;
        }
    } else if (quiet) {
        return verbosity_level::quiet;
    } else {
        return std::nullopt;
    }
}

}  // namespace plssvm::detail::cmd
