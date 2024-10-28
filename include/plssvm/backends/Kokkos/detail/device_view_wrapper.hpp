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

#include "plssvm/backends/Kokkos/detail/conditional_execution.hpp"  // PLSSVM_KOKKOS_BACKEND_INVOKE_IF_*
#include "plssvm/backends/Kokkos/execution_space.hpp"               // plssvm::kokkos::{execution_space, execution_space_to_kokkos_type_t}, plssvm::kokkos::detail::constexpr_available_execution_spaces
#include "plssvm/detail/utility.hpp"                                // plssvm::detail::unreachable

#include "Kokkos_Core.hpp"  // Kokkos::View, Kokkos::ExecutionSpace

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
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam T the value type of the underlying Kokkos::View
 * @tparam Is the indices to index the `std::array`
 */
template <typename T, std::size_t... Is>
struct create_view_variant_type_helper<T, std::index_sequence<Is...>> {
    /// The array containing all available execution spaces.
    constexpr static auto array = detail::constexpr_available_execution_spaces();
    /// The resulting variant type.
    using type = std::variant<Kokkos::View<T, execution_space_to_kokkos_type_t<array[Is]>>...>;
};

/**
 * @brief Create a `std::variant` containing all available Kokkos::View types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam T the value type of the underlying Kokkos::View
 */
template <typename T>
struct create_view_variant_type {
    /// The number of types in the final variant.
    constexpr static std::size_t N = detail::constexpr_available_execution_spaces().size();
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
     * @tparam ExecutionSpace the used Kokkos::ExecutionSpace type of the Kokkos::View
     * @param[in] view the Kokkos::View instance
     */
    template <typename ExecutionSpace>
    explicit device_view_wrapper(Kokkos::View<T, ExecutionSpace> &&view) :
        v_{ std::move(view) } { }

    /**
     * @brief Given the provided `execution_space` enum value, tries to get the `std::variant` alternative for the corresponding Kokkos::ExecutionSpace type.
     * @tparam space the `execution_space` enum value
     * @return the Kokkos::View instance (`[[nodiscard]]`)
     */
    template <execution_space space>
    [[nodiscard]] Kokkos::View<T, execution_space_to_kokkos_type_t<space>> &get() {
        return std::get<Kokkos::View<T, execution_space_to_kokkos_type_t<space>>>(v_);
    }

    /**
     * @copydoc plssvm::kokkos::detail::device_view_wrapper::get
     */
    template <execution_space space>
    [[nodiscard]] const Kokkos::View<T, execution_space_to_kokkos_type_t<space>> &get() const {
        return std::get<Kokkos::View<T, execution_space_to_kokkos_type_t<space>>>(v_);
    }

    /**
     * @brief Return the `execution_space` enum value of the currently active `std::variant` Kokkos::View type.
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
};

/**
 * @brief Given a execution @p space and the number of elements @p size, creates a Kokkos::View in the respective memory space.
 * @tparam T the value type of the underlying Kokkos::View
 * @param[in] space the specific execution space
 * @param[in] size the size of the Kokkos::View (number of elements **not** byte!)
 * @return a Kokkos::View wrapper where the active member of the internal `std::variant` corresponds to the Kokkos::View in the Kokkos::ExecutionSpace specified by @p space (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] device_view_wrapper<T> make_device_view_wrapper(const execution_space &space, const std::size_t size) {
    switch (space) {
        case execution_space::cuda:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_CUDA(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::Cuda>{ "cuda_device_ptr_view", size } };
            }));
            break;
        case execution_space::hip:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HIP(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::HIP>{ "hip_device_ptr_view", size } };
            }));
            break;
        case execution_space::sycl:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SYCL(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::SYCL>{ "sycl_device_ptr_view", size } };
            }));
            break;
        case execution_space::hpx:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_HPX(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::Experimental::HPX>{ "hpx_device_ptr_view", size } };
            }));
            break;
        case execution_space::openmp:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMP(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::OpenMP>{ "openmp_device_ptr_view", size } };
            }));
            break;
        case execution_space::openmp_target:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENMPTARGET(([&]() {
                return device_view_wrapper{ Kokkos::View<T Kokkos::OpenMPTarget>{ "openmptarget_device_ptr_view", size } };
            }));
            break;
        case execution_space::openacc:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_OPENACC(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::Experimental::OpenACC>{ "openacc_device_ptr_view", size } };
            }));
            break;
        case execution_space::threads:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_THREADS(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::Threads>{ "threads_device_ptr_view", size } };
            }));
            break;
        case execution_space::serial:
            PLSSVM_KOKKOS_BACKEND_INVOKE_IF_SERIAL(([&]() {
                return device_view_wrapper{ Kokkos::View<T, Kokkos::Serial>{ "serial_device_ptr_view", size } };
            }));
            break;
    }
    // all possible cases should be handled by the previous switch
    // -> silence missing return statement compiler warnings due to throw statement
    ::plssvm::detail::unreachable();
}

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_VIEW_WRAPPER_HPP_
