/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief A C++17 conform implementation of C++20's std::make_unique_for_overwrite.
 * @details For implementation details see: https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
 */

#ifndef PLSSVM_DETAIL_MAKE_UNIQUE_FOR_OVERWRITE_HPP_
#define PLSSVM_DETAIL_MAKE_UNIQUE_FOR_OVERWRITE_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include <cstddef>      // std::size_t
#include <cstring>      // std::memset
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::false_type, std::true_type, std::enable_if_t, std::is_array_v

namespace plssvm::detail {

/**
 * @brief Helper struct to check whether @p T is an unbounded array.
 * @tparam T the array type
 */
template <typename T>
struct is_unbounded_array : std::false_type { };

/**
 * @brief Specialization of plssvm::detail::is_unbounded_array for unbounded arrays.
 * @tparam T the array type
 */
template <typename T>
struct is_unbounded_array<T[]> : std::true_type { };  // NOLINT: see https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique

/**
 * @brief Shortcut for plssvm::detail::is_unbounded_array.
 * @tparam T the array type
 */
template <typename T>
constexpr bool is_unbounded_array_v = is_unbounded_array<T>::value;

/**
 * @brief Helper struct to check whether @p T is a bounded array.
 * @tparam T the array type
 */
template <typename T>
struct is_bounded_array : std::false_type { };

/**
 * @brief Specialization of plssvm::detail::is_bounded_array for unbounded arrays.
 * @tparam T the array type
 * @tparam N the size of the array
 */
template <typename T, std::size_t N>
struct is_bounded_array<T[N]> : std::true_type { };  // NOLINT: see https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique

/**
 * @brief Shortcut for plssvm::detail::is_bounded_array.
 * @tparam T the array type
 */
template <typename T>
constexpr bool is_bounded_array_v = is_bounded_array<T>::value;

/**
 * @brief A C++17 conform implementation of C++20's std::make_unique_for_overwrite.
 * @details For implementation details see: https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
 * @tparam T the type of the object to create
 * @return a unique pointer to the newly created object (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<std::is_array_v<T>, bool> = true>
[[nodiscard]] std::unique_ptr<T> make_unique_for_overwrite() {
    return std::unique_ptr<T>(new T);
}

/**
 * @brief A C++17 conform implementation of C++20's std::make_unique_for_overwrite.
 * @details For implementation details see: https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
 * @tparam T the type of the objects to create
 * @param[in] n the size of the array to create
 * @return a unique pointer to the newly created object (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<is_unbounded_array_v<T>, bool> = true>  // NOLINT: see https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
std::unique_ptr<T> make_unique_for_overwrite(const std::size_t n) {
    return std::unique_ptr<T>(new std::remove_extent_t<T>[n]);
}

/**
 * @brief A C++17 conform implementation of C++20's std::make_unique_for_overwrite.
 * @details For implementation details see: https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
 * @tparam T the type of the object to create
 * @tparam Args the types of the constructor arguments
 * @param[in] args the arguments to pass to the constructor
 * @return a unique pointer to the newly created object (`[[nodiscard]]`)
 */
template <typename T, typename... Args, std::enable_if_t<is_bounded_array_v<T>, bool> = true>  // NOLINT: see https://en.cppreference.com/w/cpp/memory/unique_ptr/make_unique
auto make_unique_for_overwrite(Args &&...args) = delete;

/**
 * @brief Fill the array @p dest with zeros in parallel using OpenMP if available, otherwise fall back to a sequential memset.
 * @tparam T the type of the values
 * @param[in,out] dest the array to fill with zeros
 * @param[in] count the number of values to fill
 */
template <typename T>
void parallel_zero_memset(T *dest, const std::size_t count) {
    PLSSVM_ASSERT(dest != nullptr, "The destination pointer may not be a nullptr!");

// initialize the data pointed to by dest to all zeros in parallel using OpenMP if available, otherwise fall back to a sequential memset
#if defined(_OPENMP)
    #pragma omp parallel for
    for (std::size_t i = 0; i < count; ++i) {
        dest[i] = T{ 0 };
    }
#else
    std::memset(dest, 0, count * sizeof(T));
#endif
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_MAKE_UNIQUE_FOR_OVERWRITE_HPP_
