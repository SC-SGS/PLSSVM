/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines all possible targets. Can also include targets not available on the target platform.
 */

#pragma once

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <algorithm>  // std::transform
#include <cctype>     // std::tolower
#include <istream>    // std::istream
#include <ostream>    // std::ostream
#include <string>     // std::string

namespace plssvm {

/**
 * @brief Enum class for the different targets.
 */
enum class target_platform {
    /** The default target with respect to the used backend type. Checks for available devices in the following order: NVIDIA GPUs -> AMD GPUs -> Intel GPUs -> CPUs */
    automatic,
    /** Target CPUs. */
    cpu,
    /** Target GPUs from NVIDIA */
    gpu_nvidia,
    /** Target GPUs from AMD */
    gpu_amd,
    /** Target GPUs from Intel */
    gpu_intel
};

/**
 * @brief Stream-insertion-operator overload for convenient printing of the target platform @p target.
 * @param[inout] out the output-stream to write the target platform to
 * @param[in] target the target platform
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const target_platform target) {
    switch (target) {
        case target_platform::cpu:
            return out << "cpu";
        case target_platform::gpu_nvidia:
            return out << "gpu_nvidia";
        case target_platform::gpu_amd:
            return out << "gpu_amd";
        case target_platform::gpu_intel:
            return out << "gpu_intel";
        default:
            return out << "unknown";
    }
}

/**
 * @brief Stream-extraction-operator overload for convenient converting a string to a target platform.
 * @param[inout] in input-stream to extract the target platform from
 * @param[in] target the target platform
 * @return the input-stream
 */
inline std::istream &operator>>(std::istream &in, target_platform &target) {
    std::string str;
    in >> str;
    std::transform(str.begin(), str.end(), str.begin(), [](const char c) { return std::tolower(c); });

    if (str == "cpu") {
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