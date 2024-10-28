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

#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"  // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"        // plssvm::kokkos::execution_space
#include "plssvm/detail/type_traits.hpp"                     // PLSSVM_REQUIRES
#include "plssvm/target_platforms.hpp"                       // plssvm::target_platform

#include "Kokkos_Core.hpp"  // Kokkos::ExecutionSpace::fence

#include <map>          // std::map
#include <string>       // std::string
#include <type_traits>  // std::disjunction, std::is_same
#include <variant>      // std::variant
#include <vector>       // std::vector

namespace plssvm::kokkos::detail {

namespace impl {

/**
 * @brief Uninstantiated base type for the check whether a type @p appears in a std::variant @p Variant.
 * @tparam T the type to check for inclusion
 * @tparam Variant the std::variant that should include the type @p T
 */
template <typename T, typename Variant>
struct is_type_in_variant;

/**
 * @brief Implement the inclusion check using `std::disjunction`.
 * @tparam T the type to check for inclusion
 * @tparam Variant the std::variant that should include the type @p T
 */
template <typename T, typename... Types>
struct is_type_in_variant<T, std::variant<Types...>> : std::disjunction<std::is_same<T, Types>...> { };

/**
 * @copydoc plssvm::kokkos::detail::impl::is_type_in_variant
 */
template <typename T, typename Variant>
inline constexpr bool is_type_in_variant_v = is_type_in_variant<T, Variant>::value;

}  // namespace impl

/**
 * @brief Return a `std::map` containing a mapping from all available target platforms to the available Kokkos::ExecutionSpace that supports said target platform.
 * @details If a target platform is supported by multiple Kokkos::ExecutionSpace, the order is determined by the order as returned by `list_available_execution_spaces`.
 * @return the mapping of all available target_platform <-> Kokkos::ExecutionSpace combinations (`[[nodiscard]]`)
 */
[[nodiscard]] std::map<target_platform, std::vector<execution_space>> available_target_platform_to_execution_space_mapping();

/**
 * @brief Get the name of the device represented by the `device_wrapper` @p dev.
 * @param[in] dev the device wrapper
 * @return the device name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_device_name(const device_wrapper &dev);

/**
 * @brief Wait for all kernel and/or other operations on the device wrapper in the @p dev to finish.
 * @param[in] dev the device wrapper
 */
void device_synchronize(const device_wrapper &dev);

/**
 * @brief Wait for all kernel and/or other operations on the device represented by the Kokkos::ExecutionSpace @p exec to finish.
 * @tparam ExecutionSpace the type of the Kokkos::ExecutionSpace
 * @param[in] exec the device represented by a Kokkos::ExecutionSpace
 */
template <typename ExecutionSpace, PLSSVM_REQUIRES(impl::is_type_in_variant_v<ExecutionSpace, typename impl::create_device_variant_type::type>)>
void device_synchronize(const ExecutionSpace &exec) {
    exec.fence();
}

/**
 * @brief Get the used Kokkos library version.
 * @return the library version (`[[nodiscard]]`)
 */
[[nodiscard]] std::string get_kokkos_version();

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_UTILITY_HPP_
