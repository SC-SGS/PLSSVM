/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions related to the command line parser functionality.
 */

#ifndef PLSSVM_DETAIL_CMD_UTILITY_HPP_
#define PLSSVM_DETAIL_CMD_UTILITY_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type
#include "plssvm/backends/Kokkos/execution_spaces.hpp"     // plssvm::kokkos::execution_space
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernels
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::implementation_type
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform
#include "plssvm/verbosity_levels.hpp"                     // plssvm::verbosity_level

#include "cxxopts.hpp"  // cxxopts::ParseResult, cxxopts::Options

#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <string>    // std::string
#include <utility>   // std::pair
#include <vector>    // std::vector

namespace plssvm::detail::cmd {

/**
 * @brief Filter the provided command line options starting with the @p prefix_filter.
 * @details Currently, per default filters out all options starting with "--hpx:" and "--kokkos-".
 * @attention **ONLY** single command line options are supported! I.e., "--hpx:threads=42" is supported, but not "--hpx:threads 42".
 * @param[in] argc the number of provided command line options to be filtered
 * @param[in] argv the command line options to be filtered
 * @param[in] prefix_filter a list of prefixes that should be filtered
 * @return a `std::vector` containing all non-filtered command line options (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<char *> filter_argv(int argc, char **argv, const std::vector<std::string> &prefix_filter = { "--hpx:", "--kokkos-" });

/**
 * @brief Assemble a more detailed help message for the kernel function types also containing their mathematical formula.
 * @return the kernel functions' help message (`[[nodiscard]]`)
 */
[[nodiscard]] std::string kernel_type_help_message();

/**
 * @brief If a SYCL backend is available, parse the SYCL specific command line options "--sycl_data_parallel_kernel" and "--sycl_implementation_type".
 * @details If a SYCL backend is available, returns the two parsed command line options wrapped in a `std::pair`, otherwise returns a `std::nullopt`.
 * @param[in] result the cxxopts parser result encapsulating the command line options
 * @param[in] comm the MPI communicator
 * @param[in] backend the requested backend
 * @param[in] target the requested target platform
 * @return the parsed, SYCL specific command line options (`[[nodiscard]]`)
 */
[[nodiscard]] std::optional<std::pair<sycl::data_parallel_kernel, sycl::implementation_type>> parse_and_check_sycl_options_if_available(const cxxopts::ParseResult &result, const mpi::communicator &comm, backend_type backend, target_platform target);

/**
 * @brief If the Kokkos backend is available, parse the Kokkos specific command line option "--kokkos_execution_space".
 * @details If the Kokkos backend is available, returns the parsed command line option, otherwise returns a `std::nullopt`.
 * @param[in] result the cxxopts parser result encapsulating the command line option
 * @param[in] comm the MPI communicator
 * @param[in] backend the requested backend
 * @param[in] target the requested target platform
 * @return the parsed, Kokkos specific command line option (`[[nodiscard]]`)
 */
[[nodiscard]] std::optional<kokkos::execution_space> parse_and_check_kokkos_options_if_available(const cxxopts::ParseResult &result, const mpi::communicator &comm, backend_type backend, target_platform target);

/**
 * @brief If MPI is available, parse the MPI specific command line option "--mpi_load_balancing_weights".
 * @details If MPI is available, returns the parsed command line option, otherwise returns a `std::nullopt`.
 * @param[in] result the cxxopts parser result encapsulating the command line option
 * @param[in] options all supported command line options
 * @param[in] comm the MPI communicator
 * @return the parsed, MPI specific command line option (`[[nodiscard]]`)
 */
[[nodiscard]] std::optional<std::vector<std::size_t>> parse_and_check_mpi_options_if_available(const cxxopts::ParseResult &result, const cxxopts::Options &options, const mpi::communicator &comm);

/**
 * @brief Parse the verbosity command line option.
 * @details If it was provided, returns the parsed value, otherwise returns a `std::nullopt`.
 * @param[in] result the cxxopts parser result encapsulating the command line option
 * @param[in] comm the MPI communicator
 * @return the parsed verbosity command line option (`[[nodiscard]]`)
 */
[[nodiscard]] std::optional<verbosity_level> parse_verbosity(const cxxopts::ParseResult &result, const mpi::communicator &comm);

}  // namespace plssvm::detail::cmd

#endif  // PLSSVM_DETAIL_CMD_UTILITY_HPP_
