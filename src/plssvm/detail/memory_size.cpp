/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/memory_size.hpp"

#include "fmt/core.h"  // fmt::format

#include <ostream>  // std::ostream

namespace plssvm::detail {

std::ostream &operator<<(std::ostream &out, const memory_size mem) {
    using namespace literals;
    const auto val = static_cast<long double>(mem.size_in_bytes_);
    if (mem >= 1.0_TiB) {
        out << fmt::format("{:.2f} TiB", val / 1024L / 1024L / 1024L / 1024L);
    } else if (mem >= 1.0_GiB) {
        out << fmt::format("{:.2f} GiB", val / 1024L / 1024L / 1024L);
    } else if (mem >= 1.0_MiB) {
        out << fmt::format("{:.2f} MiB", val / 1024L / 1024L);
    } else if (mem >= 1.0_KiB) {
        out << fmt::format("{:.2f} KiB", val / 1024L);
    } else {
        out << fmt::format("{} B", mem.size_in_bytes_);
    }
    return out;
}

}  // namespace plssvm::detail