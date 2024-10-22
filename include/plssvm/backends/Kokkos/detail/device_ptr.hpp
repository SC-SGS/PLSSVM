/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around a Kokkos view.
 */

#ifndef PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_PTR_HPP_
#define PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_PTR_HPP_
#pragma once

#include "plssvm/backends/gpu_device_ptr.hpp"          // plssvm::detail::gpu_device_ptr
#include "plssvm/backends/Kokkos/detail/typedefs.hpp"  // plssvm::kokkos::detail::device_view_type
#include "plssvm/shape.hpp"                            // plssvm::shape

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace

#include <cstddef>  // std::size_t

namespace plssvm::kokkos::detail {

/**
 * @brief Small wrapper class around a Kokkos view together with commonly used device functions.
 * @tparam T the type of the kernel view to wrap
 */
template <typename T>
class device_ptr : public ::plssvm::detail::gpu_device_ptr<T, Kokkos::DefaultExecutionSpace, device_view_type<T>, device_ptr<T>> {
    /// The template base type of the Kokkos device_ptr class.
    using base_type = ::plssvm::detail::gpu_device_ptr<T, Kokkos::DefaultExecutionSpace, device_view_type<T>, device_ptr<T>>;

    using base_type::data_;
    using base_type::queue_;
    using base_type::shape_;

  public:
    // Be able to use overloaded base class functions.
    using base_type::copy_to_device;
    using base_type::copy_to_device_strided;
    using base_type::copy_to_host;
    using base_type::copy_to_other_device;
    using base_type::fill;
    using base_type::memset;

    using typename base_type::const_host_pointer_type;
    using typename base_type::device_pointer_type;
    using typename base_type::host_pointer_type;
    using typename base_type::queue_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    /**
     * @brief Default construct a Kokkos device_ptr with a size of 0.
     * @details Always associated with device 0.
     */
    device_ptr() = default;
    /**
     * @brief Allocates `size * sizeof(T)` bytes in the Kokkos execution space @p exec.
     * @param[in] size the number of elements represented by the device_ptr
     * @param[in] exec the associated Kokkos execution space
     */
    explicit device_ptr(size_type size, const Kokkos::DefaultExecutionSpace &exec);
    /**
     * @brief Allocates `shape.x * shape.y * sizeof(T)` bytes in the Kokkos execution space @p exec.
     * @param[in] shape the number of elements represented by the device_ptr
     * @param[in] exec the associated Kokkos execution space
     */
    explicit device_ptr(plssvm::shape shape, const Kokkos::DefaultExecutionSpace &exec);
    /**
     * @brief Allocates `(shape.x + padding.x) * (shape.y + padding.y) * sizeof(T)` bytes in the Kokkos execution space @p exec.
     * @param[in] shape the number of elements represented by the device_ptr
     * @param[in] padding the number of padding elements added to the extent values
     * @param[in] exec the associated Kokkos execution space
     */
    device_ptr(plssvm::shape shape, plssvm::shape padding, const Kokkos::DefaultExecutionSpace &exec);

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::gpu_device_ptr(const plssvm::detail::gpu_device_ptr &)
     */
    device_ptr(const device_ptr &) = delete;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::gpu_device_ptr(plssvm::detail::gpu_device_ptr &&)
     */
    device_ptr(device_ptr &&other) noexcept = default;

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::operator=(const plssvm::detail::gpu_device_ptr &)
     */
    device_ptr &operator=(const device_ptr &) = delete;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::operator=(plssvm::detail::gpu_device_ptr &&)
     */
    device_ptr &operator=(device_ptr &&other) noexcept = default;

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::~gpu_device_ptr()
     * @details Kokkos automatically frees the memory of a Kokkos::View if the View goes out of scope.
     */
    ~device_ptr() override = default;

    /**
     * @copydoc plssvm::detail::gpu_device_ptr::memset(int, size_type, size_type)
     */
    void memset(int pattern, size_type pos, size_type num_bytes) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::fill(value_type, size_type, size_type)
     */
    void fill(value_type value, size_type pos, size_type count) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::copy_to_device(const_host_pointer_type, size_type, size_type)
     */
    void copy_to_device(const_host_pointer_type data_to_copy, size_type pos, size_type count) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::copy_to_device_strided(const_host_pointer_type, std::size_t, std::size_t, std::size_t)
     */
    void copy_to_device_strided(const_host_pointer_type data_to_copy, std::size_t spitch, std::size_t width, std::size_t height) override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::copy_to_host(host_pointer_type, size_type, size_type) const
     */
    void copy_to_host(host_pointer_type buffer, size_type pos, size_type count) const override;
    /**
     * @copydoc plssvm::detail::gpu_device_ptr::copy_to_other_device(derived_gpu_device_ptr &, size_type, size_type) const
     */
    void copy_to_other_device(device_ptr &target, size_type pos, size_type count) const override;
};

extern template class device_ptr<float>;
extern template class device_ptr<double>;

}  // namespace plssvm::kokkos::detail

#endif  // PLSSVM_BACKENDS_KOKKOS_DETAIL_DEVICE_PTR_HPP_
