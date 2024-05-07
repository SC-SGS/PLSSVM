/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/memory_size.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::trim

#include "fmt/core.h"  // fmt::format

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::detail {

std::ostream &operator<<(std::ostream &out, const memory_size mem) {
    // make custom memory size literals available
    using namespace literals;
    // get the number of bytes as decimal value
    const auto val = static_cast<double>(mem.num_bytes());
    // output the value together with the correct binary memory unit suffix
    if (mem >= 1.0_TiB) {
        out << fmt::format("{:.2f} TiB", val / 1024.0 / 1024.0 / 1024.0 / 1024.0);
    } else if (mem >= 1.0_GiB) {
        out << fmt::format("{:.2f} GiB", val / 1024.0 / 1024.0 / 1024.0);
    } else if (mem >= 1.0_MiB) {
        out << fmt::format("{:.2f} MiB", val / 1024.0 / 1024.0);
    } else if (mem >= 1.0_KiB) {
        out << fmt::format("{:.2f} KiB", val / 1024.0);
    } else {
        out << fmt::format("{} B", mem.num_bytes());
    }
    return out;
}

std::istream &operator>>(std::istream &in, memory_size &mem) {
    // get the memory size as decimal value
    long double size{};
    in >> size;

    // get the memory unit, removing all unnecessary whitespaces
    std::string unit{};
    in >> unit;
    unit = detail::trim(unit);

    // convert size to bytes according to the provided unit
    if (unit == "B") {
        // noting to do, size already given in byte
    } else if (unit == "KiB") {
        size *= 1024L;
    } else if (unit == "KB") {
        size *= 1000L;
    } else if (unit == "MiB") {
        size *= 1024L * 1024L;
    } else if (unit == "MB") {
        size *= 1000L * 1000L;
    } else if (unit == "GiB") {
        size *= 1024L * 1024L;
        size *= 1024L;
    } else if (unit == "GB") {
        size *= 1000L * 1000L;
        size *= 1000L;
    } else if (unit == "TiB") {
        size *= 1024L * 1024L;
        size *= 1024L * 1024L;
    } else if (unit == "TB") {
        size *= 1000L * 1000L;
        size *= 1000L * 1000L;
    } else {
        // provided memory unit not recognized
        in.setstate(std::ios::failbit);
    }

    // create memory_size struct
    mem = memory_size{ static_cast<unsigned long long>(size) };
    return in;
}

}  // namespace plssvm::detail
