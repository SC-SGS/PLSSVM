/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A wrapper around a Kokkos::ExecutionSpace representing a single device.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_WRAPPER_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_WRAPPER_HPP_

#include "plssvm/backends/Kokkos/detail/constexpr_available_execution_spaces.hpp"  // plssvm::kokkos::detail::constexpr_available_execution_spaces
#include "plssvm/backends/Kokkos/execution_space.hpp"                              // plssvm::kokkos::execution_space
#include "plssvm/backends/Kokkos/execution_space_type_traits.hpp"                  // plssvm::kokkos::execution_space_to_kokkos_type_t
#include "plssvm/target_platforms.hpp"                                             // plssvm::target_platform

#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <functional>  // std::invoke
#include <utility>     // std::make_index_sequence, std::index_sequence, std::forward
#include <variant>     // std::variant, std::get, std::visit
#include <vector>      // std::vector

namespace plssvm::kokkos::detail {

namespace impl {

/**
 * @brief Uninstantiated base type to create a `std::variant` containing all available Kokkos::ExecutionSpace types.
 */
template <typename>
struct create_device_variant_type_helper;

/**
 * @brief Helper struct to create a `std::variant` containing all available Kokkos::ExecutionSpace types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam Is the indices to index the `std::array`
 */
template <std::size_t... Is>
struct create_device_variant_type_helper<std::index_sequence<Is...>> {
    /// The array containing all available execution spaces.
    constexpr static auto array = detail::constexpr_available_execution_spaces();
    /// The resulting variant type.
    using type = std::variant<execution_space_to_kokkos_type_t<array[Is]>...>;
};

/**
 * @brief Create a `std::variant` containing all available Kokkos::ExecutionSpace types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 */
struct create_device_variant_type {
    /// The number of types in the final variant.
    constexpr static std::size_t N = detail::constexpr_available_execution_spaces().size();
    /// The final variant type.
    using type = typename create_device_variant_type_helper<std::make_index_sequence<N>>::type;
};

}  // namespace impl

/**
 * @brief A wrapper class around a `std::variant` that contains all available Kokkos::ExecutionSpace types.
 */
class device_wrapper {
  public:
    /// The `std::variant` type containing all Kokkos::ExecutionSpace types.
    using variant_type = typename impl::create_device_variant_type::type;

    /**
     * @brief Default construct the `std::variant` wrapper.
     */
    device_wrapper() = default;

    /**
     * @brief Construct the wrapper using the provided Kokkos::ExecutionSpace instance by forwarding its value to the underlying `std::variant`.
     * @tparam ExecutionSpace the used Kokkos::ExecutionSpace type
     * @param[in] exec the Kokkos::ExecutionSpace instance
     */
    template <typename ExecutionSpace>
    explicit device_wrapper(ExecutionSpace &&exec) :
        v_{ std::forward<ExecutionSpace>(exec) } { }

    /**
     * @brief Given the provided `execution_space` enum value, tries to get the `std::variant` alternative for the corresponding Kokkos::ExecutionSpace type.
     * @tparam space the `execution_space` enum value
     * @return the Kokkos::ExecutionSpace instance (`[[nodiscard]]`)
     */
    template <execution_space space>
    [[nodiscard]] execution_space_to_kokkos_type_t<space> &get() {
        return std::get<execution_space_to_kokkos_type_t<space>>(v_);
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_wrapper::get
     */
    template <execution_space space>
    const execution_space_to_kokkos_type_t<space> &get() const {
        return std::get<execution_space_to_kokkos_type_t<space>>(v_);
    }

    /**
     * @brief Return the `execution_space` enum value of the currently active `std::variant` Kokkos::ExecutionSpace type.
     * @return the `execution_space` enum value (`[[nodiscard]]`)
     */
    [[nodiscard]] execution_space get_execution_space() const noexcept {
        return detail::constexpr_available_execution_spaces()[v_.index()];
    }

    /**
     * @brief Invoke the function @p func on the active `std::variant` member using `std::visit` internally.
     * @tparam Func the type of the function
     * @param[in] func the function to invoke
     */
    template <typename Func>
    void execute(const Func &func) {
        // clang-format off
        std::visit([&func](auto &device) {
            std::invoke(func, device);
        }, v_);
        // clang-format on
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_wrapper::execute
     */
    template <typename Func>
    void execute(const Func &func) const {
        // clang-format off
        std::visit([&func](const auto &device) {
            std::invoke(func, device);
        }, v_);
        // clang-format on
    }

    /**
     * @brief Invoke the function @p func on the active `std::variant` member using `std::visit` internally returning the result value of the function invocation.
     * @tparam Func the type of the function
     * @param[in] func the function to invoke
     * @return the return value of function @p func (`[[nodiscard]]`)
     */
    template <typename Func>
    [[nodiscard]] auto execute_and_return(const Func &func) {
        // clang-format off
        return std::visit([&func](auto &device) {
            return std::invoke(func, device);
        }, v_);
        // clang-format on
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_wrapper::execute_and_return
     */
    template <typename Func>
    [[nodiscard]] auto execute_and_return(const Func &func) const {
        // clang-format off
        return std::visit([&func](const auto &device) {
            return std::invoke(func, device);
        }, v_);
        // clang-format on
    }

    /**
     * @brief Compare two device wrappers for equality by comparing the wrapped `std::variant`s.
     * @param[in] lhs the first device wrapper
     * @param[in] rhs the second device wrapper
     * @return `true` if both underlying `std::variant`s are equal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator==(const device_wrapper &lhs, const device_wrapper &rhs) noexcept {
        return lhs.v_ == rhs.v_;
    }

    /**
     * @brief Compare two device wrappers for inequality by comparing the wrapped `std::variant`s.
     * @param[in] lhs the first device wrapper
     * @param[in] rhs the second device wrapper
     * @return `true` if both underlying `std::variant`s are unequal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator!=(const device_wrapper &lhs, const device_wrapper &rhs) noexcept {
        return !(lhs == rhs);
    }

  private:
    /// The wrapped `std::variant` type.
    variant_type v_{};
};

/**
 * @brief Get a list of all available devices in the execution @p space that are supported by the @p target platform.
 * @param[in] space the Kokkos::ExecutionSpace to retrieve the devices from
 * @param[in] target the target platform that must be supported
 * @return all devices for the @p target in the Kokkos::ExecutionSpace @p space (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<device_wrapper> get_device_list(execution_space space, target_platform target);

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_WRAPPER_HPP_
