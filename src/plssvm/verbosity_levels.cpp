/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/verbosity_levels.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::{to_lower_case, split, trim}
#include "plssvm/detail/utility.hpp"         // plssvm::detail::to_underlying

#include "fmt/format.h"  // fmt::format, fmt::join

#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm {

verbosity_level verbosity = verbosity_level::full;

std::ostream &operator<<(std::ostream &out, const verbosity_level verb) {
    if (verb == verbosity_level::quiet) {
        return out << "quiet";
    }

    std::vector<std::string_view> level_names;

    // check bit-flags
    if ((verb & verbosity_level::libsvm) != verbosity_level::quiet) {
        level_names.emplace_back("libsvm");
    }
    if ((verb & verbosity_level::timing) != verbosity_level::quiet) {
        level_names.emplace_back("timing");
    }
    if ((verb & verbosity_level::warning) != verbosity_level::quiet) {
        level_names.emplace_back("warning");
    }
    if ((verb & verbosity_level::full) != verbosity_level::quiet) {
        level_names.emplace_back("full");
    }

    if (level_names.empty()) {
        return out << "unknown";
    } else {
        return out << fmt::format("{}", fmt::join(level_names, " | "));
    }
}

std::istream &operator>>(std::istream &in, verbosity_level &verb) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    verb = verbosity_level::quiet;

    // split the input string by "|"
    const std::vector<std::string_view> verbs = detail::split(str, '|');
    for (std::string_view verb_str : verbs) {
        verb_str = detail::trim(verb_str);
        if (verb_str == "full") {
            verb |= verbosity_level::full;
        } else if (verb_str == "warning") {
            verb |= verbosity_level::warning;
        } else if (verb_str == "timing") {
            verb |= verbosity_level::timing;
        } else if (verb_str == "libsvm") {
            verb |= verbosity_level::libsvm;
        } else if (verb_str == "quiet") {
            verb = verbosity_level::quiet;
            break;
        } else {
            in.setstate(std::ios::failbit);
            break;
        }
    }

    return in;
}

verbosity_level operator|(const verbosity_level lhs, const verbosity_level rhs) {
    return static_cast<verbosity_level>(detail::to_underlying(lhs) | detail::to_underlying(rhs));
}
verbosity_level operator|=(verbosity_level &lhs, const verbosity_level rhs) {
    const verbosity_level verb = lhs | rhs;
    return lhs = verb;
}
verbosity_level operator&(const verbosity_level lhs, const verbosity_level rhs) {
    return static_cast<verbosity_level>(detail::to_underlying(lhs) & detail::to_underlying(rhs));
}
verbosity_level operator&=(verbosity_level &lhs, const verbosity_level rhs) {
    const verbosity_level verb = lhs & rhs;
    return lhs = verb;
}

}  // namespace plssvm