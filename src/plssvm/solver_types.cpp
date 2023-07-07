/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/solver_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const solver_type solving) {
    switch (solving) {
        case solver_type::automatic:
            return out << "automatic";
        case solver_type::cg_explicit:
            return out << "cg_explicit";
        case solver_type::cg_streaming:
            return out << "cg_streaming";
        case solver_type::cg_implicit:
            return out << "cg_implicit";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, solver_type &solving) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        solving = solver_type::automatic;
    } else if (str == "cg_explicit") {
        solving = solver_type::cg_explicit;
    } else if (str == "cg_streaming") {
        solving = solver_type::cg_streaming;
    } else if (str == "cg_implicit") {
        solving = solver_type::cg_implicit;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm