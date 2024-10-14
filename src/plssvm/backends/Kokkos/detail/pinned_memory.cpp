/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/Kokkos/detail/pinned_memory.hpp"

#include "plssvm/backends/host_pinned_memory.hpp"  // plssvm::detail::host_pinned_memory
#include "plssvm/exceptions/exceptions.hpp"        // plssvm::exception

#include <cstddef>    // std::size_t
#include <exception>  // std::terminate
#include <iostream>   // std::cerr, std::endl
#include <vector>     // std::vector

namespace plssvm::kokkos::detail {

template <typename T>
pinned_memory<T>::pinned_memory(const std::vector<T> &vec) :
    pinned_memory{ vec.data(), vec.size() } { }

template <typename T>
pinned_memory<T>::pinned_memory(const T *ptr, const std::size_t size) :
    ::plssvm::detail::host_pinned_memory<T>{ ptr } {
    this->pin_memory(size * sizeof(T));
}

template <typename T>
pinned_memory<T>::~pinned_memory() {
    try {
        if (is_pinned_ && ptr_ != nullptr) {
            this->unpin_memory();
        }
    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        std::terminate();
    }
}

template class pinned_memory<float>;
template class pinned_memory<double>;

}  // namespace plssvm::kokkos::detail
