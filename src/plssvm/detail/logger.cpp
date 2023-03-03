/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/logger.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying

#include "fmt/format.h"  // fmt::format, fmt::join

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const verbosity_level verb) {
    if (verb == verbosity_level::quiet) {
        return out << "quiet";
    }

    std::vector<std::string_view> level_names;

    // check bit-flags
    if ((verb & verbosity_level::libsvm) != verbosity_level::quiet) {
        level_names.push_back("libsvm");
    }
    if ((verb & verbosity_level::timing) != verbosity_level::quiet) {
        level_names.push_back("timing");
    }
    if ((verb & verbosity_level::full) != verbosity_level::quiet) {
        level_names.push_back("full");
    }

    if (level_names.empty()) {
        return out << "unknown";
    } else {
        return out << fmt::format("{}", fmt::join(level_names, " & "));
    }
}

std::istream &operator>>(std::istream &in, verbosity_level &verb) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "full") {
        verb = verbosity_level::full;
    } else if (str == "timing") {
        verb = verbosity_level::timing;
    } else if (str == "libsvm") {
        verb = verbosity_level::libsvm;
    } else if (str == "quiet") {
        verb = verbosity_level::quiet;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

verbosity_level operator|(const verbosity_level lhs, const verbosity_level rhs) {
    return static_cast<verbosity_level>(detail::to_underlying(lhs) | detail::to_underlying(rhs));
}
verbosity_level operator&(const verbosity_level lhs, const verbosity_level rhs) {
    return static_cast<verbosity_level>(detail::to_underlying(lhs) & detail::to_underlying(rhs));
}

}  // namespace plssvm