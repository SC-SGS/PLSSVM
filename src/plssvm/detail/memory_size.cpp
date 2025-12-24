/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/memory_size.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::trim

#include "fmt/format.h"  // fmt::format

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::detail {

std::ostream &operator<<(std::ostream &out, const memory_size mem) {
    // make custom memory size literals available
    using namespace literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace

    // named constants
    constexpr double ratio_base2 = 1024.0;
    constexpr memory_size one_kibi_byte = 1.0_KiB;
    constexpr memory_size one_mibi_byte = 1.0_MiB;
    constexpr memory_size one_gibi_byte = 1.0_GiB;
    constexpr memory_size one_tibi_byte = 1.0_TiB;

    // get the number of bytes as decimal value
    const auto val = static_cast<double>(mem.num_bytes());
    // output the value together with the correct binary memory unit suffix
    if (mem >= one_tibi_byte) {
        out << fmt::format("{:.2f} TiB", val / ratio_base2 / ratio_base2 / ratio_base2 / ratio_base2);
    } else if (mem >= one_gibi_byte) {
        out << fmt::format("{:.2f} GiB", val / ratio_base2 / ratio_base2 / ratio_base2);
    } else if (mem >= one_mibi_byte) {
        out << fmt::format("{:.2f} MiB", val / ratio_base2 / ratio_base2);
    } else if (mem >= one_kibi_byte) {
        out << fmt::format("{:.2f} KiB", val / ratio_base2);
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

    constexpr long double ratio_base2 = 1024L;
    constexpr long double ratio_base10 = 1000L;

    // convert size to bytes according to the provided unit
    if (unit == "B") {
        // noting to do, size already given in byte
    } else if (unit == "KiB") {
        size *= ratio_base2;
    } else if (unit == "KB") {
        size *= ratio_base10;
    } else if (unit == "MiB") {
        size *= ratio_base2 * ratio_base2;
    } else if (unit == "MB") {
        size *= ratio_base10 * ratio_base10;
    } else if (unit == "GiB") {
        size *= ratio_base2 * ratio_base2;
        size *= ratio_base2;
    } else if (unit == "GB") {
        size *= ratio_base10 * ratio_base10;
        size *= ratio_base10;
    } else if (unit == "TiB") {
        size *= ratio_base2 * ratio_base2;
        size *= ratio_base2 * ratio_base2;
    } else if (unit == "TB") {
        size *= ratio_base10 * ratio_base10;
        size *= ratio_base10 * ratio_base10;
    } else {
        // provided memory unit not recognized
        in.setstate(std::ios::failbit);
    }

    // create memory_size struct
    mem = memory_size{ static_cast<unsigned long long>(size) };
    return in;
}

}  // namespace plssvm::detail
