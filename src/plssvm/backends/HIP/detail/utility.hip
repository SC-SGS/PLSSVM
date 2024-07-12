/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/HIP/detail/utility.hip.hpp"
// main header

#include "plssvm/backends/execution_range.hpp"  // plssvm::detail::dim_type

#include "hip/hip_runtime_api.h"  // hipGetDeviceCount, hipSetDevice, hipPeekAtLastError, hipDeviceSynchronize, hipRuntimeGetVersion

#include "fmt/format.h"  // fmt::format

#include <string>  // std::string

namespace plssvm::hip::detail {

dim3 dim_type_to_native(const ::plssvm::detail::dim_type &dims) {
    return dim3{ static_cast<unsigned int>(dims.x), static_cast<unsigned int>(dims.y), static_cast<unsigned int>(dims.z) };
}

[[nodiscard]] int get_device_count() {
    int count{};
    PLSSVM_HIP_ERROR_CHECK(hipGetDeviceCount(&count))
    return count;
}

void set_device(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    PLSSVM_HIP_ERROR_CHECK(hipSetDevice(device))
}

void peek_at_last_error() {
    PLSSVM_HIP_ERROR_CHECK(hipPeekAtLastError())
}

void device_synchronize(const int device) {
    if (device < 0 || device >= static_cast<int>(get_device_count())) {
        throw backend_exception{ fmt::format("Illegal device ID! Must be in range: [0, {}) but is {}!", get_device_count(), device) };
    }
    peek_at_last_error();
    set_device(device);
    PLSSVM_HIP_ERROR_CHECK(hipDeviceSynchronize())
}

std::string get_runtime_version() {
    // get the CUDA runtime version
    int runtime_version{};
    PLSSVM_HIP_ERROR_CHECK(hipRuntimeGetVersion(&runtime_version))
    // TODO: may be different on the CUDA platform
    // parse it to a more useful string
    int major_version = runtime_version / 10'000'000;
    int minor_version = runtime_version % 10'000'000 / 100'000;
    return fmt::format("{}.{}", major_version, minor_version);
}

}  // namespace plssvm::hip::detail
