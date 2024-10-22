/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions for the Kokkos backend.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
#pragma once

#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::execution_space
#include "plssvm/target_platforms.hpp"                              // plssvm::target_platform

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace

#include <string>  // std::string
#include <vector>  // std::vector

namespace plssvm::kokkos::detail {

/**
 * @brief Given the execution @p space, determine the respective default target platform.
 * @param[in] space the Kokkos::ExecutionSpace for which the default target platform should be determined
 * @return the default target platform (`[[nodiscard]]`)
 */
[[nodiscard]] target_platform determine_default_target_platform_from_execution_space(execution_space space);

/**
 * @brief Check whether the execution @p space supports the @p target platform. Throws an `plssvm::kokkos::backend_exception` if that's not the case.
 * @param[in] space the Kokkos::ExecutionSpace to investigate
 * @param[in] target the target platform to check
 * @throws plssvm::kokkos::backend_exception if @p space doesn't support the @p target platform
 */
void check_execution_space_target_platform_combination(execution_space space, target_platform target);

/**
 * @brief Get a list of all available devices in the execution @p space that are supported by the @p target platform.
 * @param[in] space the Kokkos::ExecutionSpace to retrieve the devices from
 * @param[in] target the target platform that must be supported
 * @return all devices for the @p target in the Kokkos::ExecutionSpace @p space (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<Kokkos::DefaultExecutionSpace> get_device_list(execution_space space, target_platform target);

/**
 * @brief Get the name of the device represented by the Kokkos::ExecutionSpace @p exec in the execution @p space.
 * @param[in] space the Kokkos::ExecutionSpace
 * @param[in] exec the device
 * @return the device name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_device_name(execution_space space, const Kokkos::DefaultExecutionSpace &exec);

/**
 * @brief Wait for all kernel and/or other operations on the Kokkos::ExecutionSpace @p exec to finish
 * @param[in] exec the Kokkos::ExecutionSpace to synchronize
 */
void device_synchronize(const Kokkos::DefaultExecutionSpace &exec);

/**
 * @brief Get the used Kokkos library version.
 * @return the library version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_kokkos_version();

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
