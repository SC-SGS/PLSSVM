/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/target_platforms.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm {

std::vector<target_platform> list_available_target_platforms() {
    std::vector<target_platform> available_targets = { target_platform::automatic };
#if defined(PLSSVM_HAS_CPU_TARGET)
    available_targets.push_back(target_platform::cpu);
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    available_targets.push_back(target_platform::gpu_nvidia);
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    available_targets.push_back(target_platform::gpu_amd);
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    available_targets.push_back(target_platform::gpu_intel);
#endif
    return available_targets;
}

target_platform determine_default_target_platform(const std::vector<target_platform> &platform_device_list) {
    // check for devices in order gpu_nvidia -> gpu_amd -> gpu_intel -> cpu
    if (::plssvm::detail::contains(platform_device_list, target_platform::gpu_nvidia)) {
        return target_platform::gpu_nvidia;
    } else if (::plssvm::detail::contains(platform_device_list, target_platform::gpu_amd)) {
        return target_platform::gpu_amd;
    } else if (::plssvm::detail::contains(platform_device_list, target_platform::gpu_intel)) {
        return target_platform::gpu_intel;
    }
    return target_platform::cpu;
}

std::ostream &operator<<(std::ostream &out, const target_platform target) {
    switch (target) {
        case target_platform::automatic:
            return out << "automatic";
        case target_platform::cpu:
            return out << "cpu";
        case target_platform::gpu_nvidia:
            return out << "gpu_nvidia";
        case target_platform::gpu_amd:
            return out << "gpu_amd";
        case target_platform::gpu_intel:
            return out << "gpu_intel";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, target_platform &target) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        target = target_platform::automatic;
    } else if (str == "cpu") {
        target = target_platform::cpu;
    } else if (str == "gpu_nvidia") {
        target = target_platform::gpu_nvidia;
    } else if (str == "gpu_amd") {
        target = target_platform::gpu_amd;
    } else if (str == "gpu_intel") {
        target = target_platform::gpu_intel;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm