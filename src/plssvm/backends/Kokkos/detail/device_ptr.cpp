/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"

#include "plssvm/backends/Kokkos/detail/typedefs.hpp"  // plssvm::kokkos::detail::{device_view_type, host_view_type}
#include "plssvm/backends/Kokkos/detail/utility.hpp"   // plssvm::detail::device_synchronize
#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/detail/assert.hpp"                    // PLSSVM_ASSERT
#include "plssvm/shape.hpp"                            // plssvm::shape

#include "Kokkos_Core.hpp"  // Kokkos::DefaultExecutionSpace, Kokkos::subview, Kokkos::parallel_for, KOKKOS_LAMBDA, Kokkos::deep_copy

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <cstddef>    // std::size_t
#include <cstring>    // std::memcpy
#include <utility>    // std::make_pair
#include <vector>     // std::vector

namespace plssvm::kokkos::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const Kokkos::DefaultExecutionSpace &exec) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, exec } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const Kokkos::DefaultExecutionSpace &exec) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, exec } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const plssvm::shape padding, const Kokkos::DefaultExecutionSpace &exec) :
    base_type{ shape, padding, exec } {
    data_ = device_view_type<T>{ "device_ptr_view", this->size_padded() };
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, this->size_padded()) };
    }
    const size_type rnum_bytes = std::min(num_bytes, (this->size_padded() - pos) * sizeof(value_type));

    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + (rnum_bytes / sizeof(value_type))));
    // fill subview with constant data
    Kokkos::parallel_for("device_ptr_memset", num_bytes, KOKKOS_LAMBDA(const std::size_t idx) {
        // Cast the view's data pointer to unsigned char* (byte access)
        reinterpret_cast<unsigned char*>(data_subview.data())[idx] = pattern; });

    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");

    if (pos >= this->size_padded()) {
        throw backend_exception{ fmt::format("Illegal access in fill!: {} >= {}", pos, this->size_padded()) };
    }
    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + rcount));
    // fill subview with constant data
    Kokkos::deep_copy(data_subview, value);

    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create view of the host data
    const host_view_type<const T> host_view{ data_to_copy, rcount };
    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + rcount));
    // copy the data to the device subview
    Kokkos::deep_copy(data_subview, host_view);

    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid host pointer for the data to copy!");

    if (width > spitch) {
        throw backend_exception{ fmt::format("Invalid width and spitch combination specified (width: {} <= spitch: {})!", width, spitch) };
    }

    // TODO: strided copy to device in Kokkos currently not possible
    if (spitch == width) {
        // can use normal copy since we have no line strides
        this->copy_to_device(data_to_copy, 0, width * height);
    } else {
        std::vector<value_type> temp(this->shape_padded().x * height, value_type{ 0.0 });
        value_type *pos = temp.data();
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(pos, data_to_copy + row * spitch, width * sizeof(value_type));
            pos += this->shape_padded().x;
        }
        this->copy_to_device(temp);
    }

    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid host pointer for the data to copy!");

    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create view of the host data
    const host_view_type<T> host_view{ buffer, rcount };
    // create subview of the device data
    auto data_subview = Kokkos::subview(data_, std::make_pair(pos, pos + rcount));
    // copy the data to the host
    Kokkos::deep_copy(host_view, data_subview);

    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != device_view_type<T>{}, "Invalid data pointer! Maybe *this has been default constructed?");
    PLSSVM_ASSERT(target.get() != device_view_type<T>{}, "Invalid target pointer! Maybe target has been default constructed?");

    const size_type rcount = std::min(count, this->size_padded() - pos);
    if (target.size_padded() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform copy (needed: {}, provided: {})!", rcount, target.size_padded()) };
    }

    // TODO: use Kokkos function?
    std::vector<value_type> temp(rcount);
    this->copy_to_host(temp, pos, rcount);
    target.copy_to_device(temp);

    detail::device_synchronize(queue_);
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::kokkos::detail
