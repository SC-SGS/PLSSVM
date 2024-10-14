/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/device_ptr.hpp"

namespace plssvm::kokkos::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const int queue) :
    device_ptr{ plssvm::shape{ size, 1 }, plssvm::shape{ 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const int queue) :
    device_ptr{ shape, plssvm::shape{ 0, 0 }, queue } { }

template <typename T>
device_ptr<T>::device_ptr(const plssvm::shape shape, const plssvm::shape padding, const int queue) :
    base_type{ shape, padding, queue } {
    static std::size_t count = 0;
    // TODO: queue type, check range?
    // TODO: how to assign a view to a GPU in a multi-GPU setting?
    data_ = device_view_type<T>{ fmt::format("device_ptr_{}", count++), this->size_padded() };
}

template <typename T>
device_ptr<T>::~device_ptr() {
    // avoid compiler warnings
    try {
        // TODO:
    } catch (const plssvm::exception &e) {
        std::cout << e.what_with_loc() << std::endl;
        std::terminate();
    }
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

//    detail::set_device(queue_);  // TODO:
    const size_type rcount = std::min(count, this->size_padded() - pos);

    // create view of the host data
    host_view_type<T> host_view{ data_to_copy, rcount };
    // create subview of the device data
    device_subview_type<T> data_subview{ data_, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(pos, rcount) };  // TODO: view via typedef
    Kokkos::deep_copy(data_subview, host_view);
}

template <typename T>
void device_ptr<T>::copy_to_device_strided(const_host_pointer_type data_to_copy, const std::size_t spitch, const std::size_t width, const std::size_t height) {
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
}

template <typename T>
void device_ptr<T>::copy_to_other_device(device_ptr &target, const size_type pos, const size_type count) const {
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::kokkos::detail
