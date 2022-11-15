/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"

#include "plssvm/backends/CUDA/detail/utility.cuh"  // PLSSVM_CUDA_ERROR_CHECK, plssvm::cuda::detail::get_device_count
#include "plssvm/backends/CUDA/exceptions.hpp"      // plssvm::cuda::backend_exception
#include "plssvm/backends/gpu_device_ptr.hpp"       // plssvm::detail::gpu_device_ptr
#include "plssvm/detail/assert.hpp"                 // PLSSVM_ASSERT

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min

namespace plssvm::cuda::detail {

template <typename T>
device_ptr<T>::device_ptr(const size_type size, const queue_type device) :
    base_type{ size, device } {
    if (queue_ < 0 || queue_ >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}.", get_device_count(), queue_) };
    }
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(queue_));
    PLSSVM_CUDA_ERROR_CHECK(cudaMalloc(reinterpret_cast<void **>(&data_), size_ * sizeof(value_type)));
}

template <typename T>
device_ptr<T>::~device_ptr() {
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(queue_));
    PLSSVM_CUDA_ERROR_CHECK(cudaFree(data_));
}

template <typename T>
void device_ptr<T>::memset(const int pattern, const size_type pos, const size_type num_bytes) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(queue_));
    const size_type rnum_bytes = std::min(num_bytes, (size_ - pos) * sizeof(value_type));
    PLSSVM_CUDA_ERROR_CHECK(cudaMemset(data_ + pos, pattern, rnum_bytes));
}

template <typename value_type, typename size_type>
__global__ void gpu_fill(value_type* data, value_type value, size_type pos, size_type count) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // fill the array
    if (idx < count) {
        data[pos + idx] = value;
    }
}

template <typename T>
void device_ptr<T>::fill(const value_type value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }

    detail::set_device(queue_);

    // run GPU kernel
    const size_type rcount = std::min(count, size_ - pos);
    int block_size = 512;
    int grid_size = (rcount + block_size - 1) / block_size;
    gpu_fill<<<grid_size, block_size>>>(data_, value, pos, rcount);

    detail::peek_at_last_error();
    detail::device_synchronize(queue_);
}

template <typename T>
void device_ptr<T>::copy_to_device(const_host_pointer_type data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");
    PLSSVM_ASSERT(data_to_copy != nullptr, "Invalid pointer for the data to copy!");

    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(queue_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(data_ + pos, data_to_copy, rcount * sizeof(value_type), cudaMemcpyHostToDevice));
}

template <typename T>
void device_ptr<T>::copy_to_host(host_pointer_type buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(data_ != nullptr, "Invalid data pointer!");
    PLSSVM_ASSERT(buffer != nullptr, "Invalid pointer for the data to copy!");

    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(queue_));
    const size_type rcount = std::min(count, size_ - pos);
    PLSSVM_CUDA_ERROR_CHECK(cudaMemcpy(buffer, data_ + pos, rcount * sizeof(value_type), cudaMemcpyDeviceToHost));
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::cuda::detail