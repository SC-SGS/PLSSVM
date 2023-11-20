/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/detail/utility.cuh"

#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception

#include "fmt/format.h"  // fmt::format

namespace plssvm::cuda::detail {

void gpu_assert(const cudaError_t code) {
    if (code != cudaSuccess) {
        throw backend_exception{ fmt::format("CUDA assert '{}' ({}): {}", cudaGetErrorName(code), code, cudaGetErrorString(code)) };
    }
}

[[nodiscard]] int get_device_count() {
    int count{};
    PLSSVM_CUDA_ERROR_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void set_device(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    PLSSVM_CUDA_ERROR_CHECK(cudaSetDevice(device));
}

void peek_at_last_error() {
    PLSSVM_CUDA_ERROR_CHECK(cudaPeekAtLastError());
}

void device_synchronize(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    peek_at_last_error();
    set_device(device);
    PLSSVM_CUDA_ERROR_CHECK(cudaDeviceSynchronize());
}

}  // namespace plssvm::cuda::detail