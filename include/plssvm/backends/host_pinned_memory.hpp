/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around RAII enabled pinned memory.
 */

#ifndef PLSSVM_BACKENDS_HOST_PINNED_MEMORY_HPP_
#define PLSSVM_BACKENDS_HOST_PINNED_MEMORY_HPP_
#pragma once

#include "plssvm/detail/logging_without_performance_tracking.hpp"  // plssvm::detail::log_untracked
#include "plssvm/detail/type_list.hpp"                             // plssvm::detail::{supported_real_types, tuple_contains_v}
#include "plssvm/matrix.hpp"                                       // plssvm::matrix, plssvm::layout_type
#include "plssvm/verbosity_levels.hpp"                             // plssvm::verbosity_level

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::detail {

/**
 * @brief A small RAII wrapper class to register/unregister pinned memory.
 * @tparam T the type of the data array that should be pinned
 */
template <typename T>
class host_pinned_memory {
    // make sure only valid template types are used
    static_assert(detail::tuple_contains_v<T, detail::supported_real_types>, "Illegal real type provided! See the 'real_type_list' in the type_list.hpp header for a list of the allowed types.");

  public:
    /// The type of the values used in the pinned memory.
    using value_type = T;

    /**
     * @brief Must provide a memory that should be pinned.
     */
    host_pinned_memory() = delete;
    /**
     * @brief Delete the copy-constructor.
     */
    host_pinned_memory(const host_pinned_memory &) = delete;
    /**
     * @brief Delete the move-constructor.
     */
    host_pinned_memory(host_pinned_memory &&) noexcept = delete;
    /**
     * @brief Delete the copy-assignment operator.
     * @return `*this`
     */
    host_pinned_memory &operator=(const host_pinned_memory &) = delete;
    /**
     * @brief Delete the move-assignment operator.
     * @return `*this`
     */
    host_pinned_memory &operator=(host_pinned_memory &&) noexcept = delete;

    /**
     * @brief Register the memory managed by the pointer @p ptr with @p size to use pinned memory.
     * @param[in] ptr the memory to pin
     */
    explicit host_pinned_memory(const T *ptr);

    /**
     * @brief Unregister the pinned memory.
     */
    virtual ~host_pinned_memory() = 0;

    /**
     * @brief Return whether the memory could be pinned or not.
     * @return `true` if the memory could be pinned, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_pinned() const noexcept { return is_pinned_; }

  protected:
    /**
     * @brief Pin @p num_bytes of the memory.
     * @param[in] num_bytes the number of bytes
     */
    virtual void pin_memory(std::size_t num_bytes);
    /**
     * @brief Unpin the memory.
     */
    virtual void unpin_memory();

    /// The pointer to the memory that should be pinned.
    const T *ptr_{ nullptr };
    /// `true` if the memory could be pinned, `false` otherwise.
    bool is_pinned_{ false };
};

template <typename T>
host_pinned_memory<T>::host_pinned_memory(const T *ptr) :
    ptr_{ ptr } {
}

template <typename T>
host_pinned_memory<T>::~host_pinned_memory() = default;

template <typename T>
void host_pinned_memory<T>::pin_memory(const std::size_t) {
    // explicitly set flag
    is_pinned_ = false;
}

template <typename T>
void host_pinned_memory<T>::unpin_memory() {
    // explicitly set flag
    is_pinned_ = false;
}

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_HOST_PINNED_MEMORY_HPP_
