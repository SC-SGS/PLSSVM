/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/detail/utility.hip.hpp"

#include "plssvm/backends/HIP/exceptions.hpp"  // plssvm::hip::backend_exception

#include "hip/hip_runtime_api.h"  // hipError_t, hipSuccess, hipGetErrorName, hipGetErrorString, hipGetDeviceCount, hipSetDevice, hipPeekAtLastError, hipDeviceSynchronize

#include "fmt/core.h"  // fmt::format

namespace plssvm::hip::detail {

void gpu_assert(const hipError_t code) {
    if (code != hipSuccess) {
        throw backend_exception{ fmt::format("HIP assert '{}' ({}): {}", hipGetErrorName(code), code, hipGetErrorString(code)) };
    }
}

[[nodiscard]] int get_device_count() {
    int count;
    PLSSVM_HIP_ERROR_CHECK(hipGetDeviceCount(&count));
    return count;
}

void set_device(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device));
}

void peek_at_last_error() {
    PLSSVM_HIP_ERROR_CHECK(hipPeekAtLastError());
}

void device_synchronize(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    peek_at_last_error();
    set_device(device);
    PLSSVM_HIP_ERROR_CHECK(hipDeviceSynchronize());
}

}  // namespace plssvm::hip::detail
