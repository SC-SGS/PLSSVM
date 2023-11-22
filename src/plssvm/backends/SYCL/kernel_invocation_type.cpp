/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::sycl {

std::ostream &operator<<(std::ostream &out, const kernel_invocation_type target) {
    switch (target) {
        case kernel_invocation_type::automatic:
            return out << "automatic";
        case kernel_invocation_type::nd_range:
            return out << "nd_range";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, kernel_invocation_type &target) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        target = kernel_invocation_type::automatic;
    } else if (str == "nd_range") {
        target = kernel_invocation_type::nd_range;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::sycl