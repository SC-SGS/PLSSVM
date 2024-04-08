/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/detail/pinned_memory.cuh"

#include "plssvm/backends/CUDA/detail/utility.cuh"  // PLSSVM_CUDA_ERROR_CHECK
#include "plssvm/backends/host_pinned_memory.hpp"   // plssvm::detail::host_pinned_memory
#include "plssvm/detail/assert.hpp"                 // PLSSVM_ASSERT
#include "plssvm/detail/performance_tracker.hpp"    // PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
#include "plssvm/exceptions/exceptions.hpp"         // plssvm::exception

#include "cuda_runtime_api.h"  // cudaHostRegister

#include "driver_types.h"  // cudaHostRegisterDefault
#include <chrono>          // std::chrono::steady_clock::{now, time_point}, std::chrono::duration_cast
#include <cstddef>         // std::size_t
#include <exception>       // std::terminate
#include <iostream>        // std::cerr, std::endl
#include <vector>          // std::vector

namespace plssvm::cuda::detail {

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

template <typename T>
void pinned_memory<T>::pin_memory(const std::size_t num_bytes) {
    PLSSVM_ASSERT(num_bytes > 0, "Can't pin a 0 B memory!");
    PLSSVM_ASSERT(ptr_ != nullptr, "ptr_ may not be the nullptr!");

    [[maybe_unused]] const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    // register memory to be pinned
    PLSSVM_CUDA_ERROR_CHECK(cudaHostRegister((void *) ptr_, num_bytes, cudaHostRegisterDefault));
    // set flag
    is_pinned_ = true;

    [[maybe_unused]] const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking_entry{ "kernel_matrix", "pin_memory_runtime", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) }));
}

template <typename T>
void pinned_memory<T>::unpin_memory() {
    PLSSVM_CUDA_ERROR_CHECK(cudaHostUnregister((void *) ptr_));
    // set flag
    is_pinned_ = false;
}

template class pinned_memory<float>;
template class pinned_memory<double>;

}  // namespace plssvm::cuda::detail
