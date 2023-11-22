/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Small wrapper around a SYCL device pointer for the DPC++ SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_DEVICE_PTR_HPP_
#define PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_DEVICE_PTR_HPP_
#pragma once

#include "plssvm/backends/SYCL/DPCPP/detail/queue.hpp"  // plssvm::sycl::detail::queue (PImpl)
#include "plssvm/backends/gpu_device_ptr.hpp"           // plssvm::detail::gpu_device_ptr

#include <array>  // std::array

namespace plssvm::dpcpp::detail {

/**
 * @brief Small wrapper class around a DPC++ device pointer together with commonly used device functions.
 * @tparam T the type of the kernel pointer to wrap
 */
template <typename T>
class device_ptr : public ::plssvm::detail::gpu_device_ptr<T, queue> {
    /// The template base type of the DPC++ device_ptr class.
    using base_type = ::plssvm::detail::gpu_device_ptr<T, queue>;

    using base_type::data_;
    using base_type::queue_;
    using base_type::extents_;

  public:
    // Be able to use overloaded base class functions.
    using base_type::memset;
    using base_type::fill;
    using base_type::copy_to_device;
    using base_type::copy_to_host;

    using typename base_type::const_host_pointer_type;
    using typename base_type::device_pointer_type;
    using typename base_type::host_pointer_type;
    using typename base_type::queue_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    /**
     * @brief Default construct a device_ptr with a size of 0.
     */
    device_ptr() = default;
    /**
     * @brief Allocates `size * sizeof(T)` bytes on the device associated with @p q.
     * @param[in] size the number of elements represented by the device_ptr
     * @param[in] q the associated SYCL queue
     */
    device_ptr(size_type size, const queue &q);
    /**
     * @brief Allocates `extents[0] * extents[1] * sizeof(T)` bytes on the device associated with @p q.
     * @param[in] extents the number of elements represented by the device_ptr
     * @param[in] q the associated SYCL queue
     */
    device_ptr(std::array<size_type, 2> extents, const queue &q);
    /**
     * @brief Allocates `(extents[0] + padding[0]) * (extents[1] * padding[1]) * sizeof(T)` bytes on the device associated with @p q.
     * @param[in] extents the number of elements represented by the device_ptr
     * @param[in] padding the number of padding elements added to the extent values
     * @param[in] q the associated SYCL queue
     */
    device_ptr(std::array<size_type, 2> extents, std::array<size_type, 2> padding, const queue &q);

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
     */
    ~device_ptr() override;

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
     * @copydoc plssvm::detail::gpu_device_ptr::copy_to_host(host_pointer_type, size_type, size_type) const
     */
    void copy_to_host(host_pointer_type buffer, size_type pos, size_type count) const override;
};

extern template class device_ptr<float>;
extern template class device_ptr<double>;

}  // namespace plssvm::dpcpp::detail

#endif  // PLSSVM_BACKENDS_SYCL_DPCPP_DETAIL_DEVICE_PTR_HPP_