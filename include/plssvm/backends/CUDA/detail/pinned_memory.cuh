/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around RAII enabled cudaHostRegister.
 */

#ifndef PLSSVM_BACKENDS_CUDA_DETAIL_PINNED_MEMORY_HPP_
#define PLSSVM_BACKENDS_CUDA_DETAIL_PINNED_MEMORY_HPP_
#pragma once

#include "plssvm/backends/host_pinned_memory.hpp"  // plssvm::detail::host_pinned_memory
#include "plssvm/matrix.hpp"                       // plssvm::matrix, plssvm::layout_type

#include <cstddef>  // std::size_t
#include <vector>   // std::vector

namespace plssvm::cuda::detail {

/**
 * @brief A small RAII wrapper class to register/unregister pinned memory.
 * @tparam T the type of the data array that should be pinned
 */
template <typename T>
class [[nodiscard]] pinned_memory final : public ::plssvm::detail::host_pinned_memory<T> {
    /// The template base type of the CUDA pinned_memory class.
    using base_type = ::plssvm::detail::host_pinned_memory<T>;

    using base_type::is_pinned_;
    using base_type::ptr_;

  public:
    using typename base_type::value_type;

    /**
     * @brief Register the memory managed by the matrix @p matr to use pinned memory.
     * @tparam layout the layout type of the matrix
     * @param[in] matr the memory to pin
     */
    template <layout_type layout>
    explicit pinned_memory(const matrix<T, layout> &matr) :
        pinned_memory{ matr.data(), matr.size_padded() } { }

    /**
     * @brief Register the memory managed by the vector @p vec to use pinned memory.
     * @param[in] vec the memory to pin
     */
    explicit pinned_memory(const std::vector<T> &vec);
    /**
     * @brief Register the memory managed by the pointer @p ptr with @p size to use pinned memory.
     * @param[in] ptr the memory to pin
     * @param[in] size the number of elements in the memory region to pin (**not** bytes!)
     */
    pinned_memory(const T *ptr, std::size_t size);
    /**
     * @brief Unregister the memory managed by this object.
     */
    ~pinned_memory() override;

  private:
    /**
     * @copydetails plssvm::detail::host_pinned_memory::pin_memory
     */
    void pin_memory(std::size_t num_bytes) override;
    /**
     * @copydetails plssvm::detail::host_pinned_memory::unpin_memory
     */
    void unpin_memory() override;
};

extern template class pinned_memory<float>;
extern template class pinned_memory<double>;

}  // namespace plssvm::cuda::detail

#endif  // PLSSVM_BACKENDS_CUDA_DETAIL_PINNED_MEMORY_HPP_
