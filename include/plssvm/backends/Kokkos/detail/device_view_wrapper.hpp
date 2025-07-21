/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A wrapper around a Kokkos::View.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_VIEW_WRAPPER_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_VIEW_WRAPPER_HPP_

#include "plssvm/backends/Kokkos/detail/constexpr_available_memory_spaces.hpp"  // plssvm::kokkos::detail::constexpr_available_memory_spaces
#include "plssvm/backends/Kokkos/detail/device_wrapper.hpp"                     // plssvm::kokkos::detail::device_wrapper
#include "plssvm/backends/Kokkos/execution_space.hpp"                           // plssvm::kokkos::execution_space
#include "plssvm/backends/Kokkos/memory_space.hpp"                              // plssvm::kokkos::memory_space
#include "plssvm/backends/Kokkos/memory_space_type_traits.hpp"                  // plssvm::kokkos::{memory_space_to_kokkos_type_t, kokkos_execution_space_to_kokkos_memory_space_t}
#include "plssvm/detail/type_traits.hpp"                                        // plssvm::detail::remove_cvref_t

#include "Kokkos_Core.hpp"  // Kokkos::View, Kokkos::MemorySpace

#include <array>       // std::array
#include <cstddef>     // std::size_t
#include <functional>  // std::invoke
#include <utility>     // std::make_index_sequence, std::index_sequence, std::move
#include <variant>     // std::variant, std::get, std::visit

namespace plssvm::kokkos::detail {

namespace impl {

/**
 * @brief Uninstantiated base type to create a `std::variant` containing all available Kokkos::View types.
 */
template <typename, typename>
struct create_view_variant_type_helper;

/**
 * @brief Helper struct to create a `std::variant` containing all available Kokkos::View types by iterating over the `std::array` of
 *        `plssvm::kokkos::memory_space` values as returned by `plssvm::kokkos::detail::constexpr_available_memory_spaces()`.
 * @tparam T the value type of the underlying Kokkos::View
 * @tparam Is the indices to index the `std::array`
 */
template <typename T, std::size_t... Is>
struct create_view_variant_type_helper<T, std::index_sequence<Is...>> {
    /// The array containing all available memory spaces.
    constexpr static auto array = detail::constexpr_available_memory_spaces();
    /// The resulting variant type.
    using type = std::variant<Kokkos::View<T, memory_space_to_kokkos_type_t<array[Is]>>...>;
};

/**
 * @brief Create a `std::variant` containing all available Kokkos::View types by iterating over the `std::array` of
 *        `plssvm::kokkos::memory_space` values as returned by `plssvm::kokkos::detail::constexpr_available_memory_spaces()`.
 * @tparam T the value type of the underlying Kokkos::View
 */
template <typename T>
struct create_view_variant_type {
    /// The number of types in the final variant.
    constexpr static std::size_t N = detail::constexpr_available_memory_spaces().size();
    /// The final variant type.
    using type = typename create_view_variant_type_helper<T, std::make_index_sequence<N>>::type;
};

}  // namespace impl

/**
 * @brief A wrapper class around a `std::variant` that contains all available Kokkos::View types.
 * @tparam T the value type of the underlying Kokkos::View
 */
template <typename T>
class device_view_wrapper {
  public:
    /// The `std::variant` type containing all Kokkos::View types.
    using variant_type = typename impl::create_view_variant_type<T>::type;

    /**
     * @brief Default construct the `std::variant` wrapper.
     */
    device_view_wrapper() = default;

    /**
     * @brief Construct the wrapper using the provided Kokkos::View instance by forwarding its value to the underlying `std::variant`.
     * @tparam MemorySpace the used Kokkos::MemorySpace type of the Kokkos::View
     * @param[in] view the Kokkos::View instance
     */
    template <typename MemorySpace>
    explicit device_view_wrapper(Kokkos::View<T, MemorySpace> &&view, const bool use_usm_allocations = false) :
        v_{ std::move(view) },
        use_usm_allocations_{ use_usm_allocations } { }

    /**
     * @brief Given the provided `execution_space` enum value, tries to get the `std::variant` alternative for the corresponding Kokkos::ExecutionSpace type.
     * @tparam space the `execution_space` enum value
     * @tparam use_usm_allocations if `true` use USM allocations
     * @return the Kokkos::View instance (`[[nodiscard]]`)
     */
    template <execution_space space, bool use_usm_allocations = false>
    [[nodiscard]] auto &get() {
        constexpr memory_space mem_space = execution_space_to_memory_space_v<space, use_usm_allocations>;
        return std::get<Kokkos::View<T, memory_space_to_kokkos_type_t<mem_space>>>(v_);
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_view_wrapper::get
     */
    template <execution_space space, bool use_usm_allocations = false>
    [[nodiscard]] const auto &get() const {
        constexpr memory_space mem_space = execution_space_to_memory_space_v<space, use_usm_allocations>;
        return std::get<Kokkos::View<T, memory_space_to_kokkos_type_t<mem_space>>>(v_);
    }

    /**
     * @brief Return the `memory_space` enum value of the currently active `std::variant` Kokkos::View type.
     * @return the `memory_space` enum value (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr memory_space get_memory_space() const noexcept {
        return detail::constexpr_available_memory_spaces()[v_.index()];
    }

    /**
     * @brief Check whether USM allocations are used.
     * @return `true` if USM allocations are used, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool uses_usm_allocations() const noexcept {
        return use_usm_allocations_;
    }

    /**
     * @brief Invoke the function @p func on the active `std::variant` member using `std::visit` internally.
     * @tparam Func the type of the function
     * @param[in] func the function to invoke
     */
    template <typename Func>
    void execute(const Func &func) {
        // clang-format off
        std::visit([&func](auto &view) {
            std::invoke(func, view);
        }, v_);
        // clang-format on
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_view_wrapper::execute
     */
    template <typename Func>
    void execute(const Func &func) const {
        // clang-format off
        std::visit([&func](const auto &view) {
            std::invoke(func, view);
        }, v_);
        // clang-format on
    }

    /**
     * @brief Compare two device view wrappers for equality by comparing the wrapped `std::variant`s.
     * @param[in] lhs the first device view wrapper
     * @param[in] rhs the second device view wrapper
     * @return `true` if both underlying `std::variant`s are equal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator==(const device_view_wrapper &lhs, const device_view_wrapper &rhs) noexcept {
        return lhs.v_ == rhs.v_;
    }

    /**
     * @brief Compare two device view wrappers for inequality by comparing the wrapped `std::variant`s.
     * @param[in] lhs the first device view wrapper
     * @param[in] rhs the second device view wrapper
     * @return `true` if both underlying `std::variant`s are unequal, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] friend bool operator!=(const device_view_wrapper &lhs, const device_view_wrapper &rhs) noexcept {
        return !(lhs == rhs);
    }

  private:
    /// The wrapped `std::variant` type.
    variant_type v_;
    /// `true` if USM allocations and, therefore, other Kokkos::MemorySpaces, are used.
    bool use_usm_allocations_;
};

/**
 * @brief Given a execution @p space and the number of elements @p size, creates a Kokkos::View in the respective memory space.
 * @tparam T the value type of the underlying Kokkos::View
 * @param[in] device the device for which this view should be allocated
 * @param[in] size the size of the Kokkos::View (number of elements **not** byte!)
 * @param[in] use_usm_allocations decide whether a USM memory space should be used or not
 * @return a Kokkos::View wrapper where the active member of the internal `std::variant` corresponds to the Kokkos::View in the Kokkos::MemorySpace based on the requested Kokkos::ExecutionSpace and @p use_usm_allocations (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] device_view_wrapper<T> make_device_view_wrapper(const device_wrapper &device, const std::size_t size, const bool use_usm_allocations) {
    return device.execute_and_return([&](const auto &value) {
        // get the Kokkos execution space
        using kokkos_execution_space_type = ::plssvm::detail::remove_cvref_t<decltype(value)>;
        // check whether we want to use USM allocations or not
        if (use_usm_allocations) {
            return device_view_wrapper{ Kokkos::View<T, kokkos_execution_space_to_kokkos_memory_space_t<kokkos_execution_space_type, true>>{ Kokkos::view_alloc(value, "usm_device_ptr_view"), size }, use_usm_allocations };
        } else {
            return device_view_wrapper{ Kokkos::View<T, kokkos_execution_space_to_kokkos_memory_space_t<kokkos_execution_space_type, false>>{ Kokkos::view_alloc(value, "device_ptr_view"), size }, use_usm_allocations };
        }
    });
}

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_VIEW_WRAPPER_HPP_
