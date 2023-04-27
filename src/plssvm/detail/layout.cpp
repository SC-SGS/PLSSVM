/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/layout.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>                               // std::ios::failbit
#include <istream>                           // std::istream
#include <ostream>                           // std::ostream
#include <string>                            // std::string

namespace plssvm::detail {

std::ostream &operator<<(std::ostream &out, const layout_type layout) {
    switch (layout) {
        case layout_type::aos:
            return out << "Array-of-Structs";
        case layout_type::soa:
            return out << "Struct-of-Arrays";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, layout_type &layout) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "aos" || str == "array-of-structs") {
        layout = layout_type::aos;
    } else if (str == "soa" || str == "struct-of-arrays") {
        layout = layout_type::soa;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::detail