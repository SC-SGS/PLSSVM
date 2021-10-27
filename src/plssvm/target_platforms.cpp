/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/target_platforms.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

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