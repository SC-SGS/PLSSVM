/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"

#include "plssvm/backends/Kokkos/exceptions.hpp"  // plssvm::kokkos::backend_exception
#include "plssvm/detail/assert.hpp"               // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/shape.hpp"                       // plssvm::shape

#include "Kokkos_Core.hpp"

#include "fmt/core.h"  // fmt::format

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cout, std::endl
#include <vector>     // std::vector

namespace plssvm::kokkos::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const Kokkos::DefaultExecutionSpace exec) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, exec } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const Kokkos::DefaultExecutionSpace exec) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, exec } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const plssvm::shape padding, const Kokkos::DefaultExecutionSpace exec) :
    base_type{ shape, padding, exec } {
    // TODO: GUARD behind ifdef!
    data_ = device_view_type<T>{ fmt::format("device_{}_view", exec.cuda_device()), this->size_padded() };
}

template <typename T>
device_ptr<T>::~device_ptr() {
    // Kokkos automatically frees the memory of a Kokkos::View if the View goes out of scope
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create view of the host data
    const host_view_type<const T> host_view{ data_to_copy, rcount };
    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + rcount));
    // copy the data to the device subview
    Kokkos::deep_copy(data_subview, host_view);
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    if (width > spitch) {
        throw backend_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }

    Kokkos::View<T **, Kokkos::LayoutRight> view_2d{ data_.data(), this->shape_padded().x, this->shape_padded().y };
    // TODO: implement
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create view of the host data
    const host_view_type<T> host_view{ buffer, rcount };
    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + rcount));
    // copy the data to the host
    Kokkos::deep_copy(host_view, data_subview);
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != view_type<T>{}, "Invalid target pointer! Maybe target has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }

    // TODO: use Kokkos function?
    std::vector<value_type> temp(rcount);
    this->copy_to_host(temp, pos, rcount);
    target.copy_to_device(temp);
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::kokkos::detail
