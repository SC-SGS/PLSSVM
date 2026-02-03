/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/detail/jit_info.hpp"

#include "fmt/chrono.h"  // NOLINT(misc-include-cleaner): format std::chrono types
#include "fmt/format.h"  // fmt::format

#include <chrono>   // NOLINT(misc-include-cleaner): std::chrono::milliseconds
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm::opencl::detail {

std::ostream &operator<<(std::ostream &out, const jit_info::caching_status status) {
    switch (status) {
        case jit_info::caching_status::success:
            return out << "success";
        case jit_info::caching_status::error_no_cached_files:
            return out << "no cached files exist (checksum missmatch)";
        case jit_info::caching_status::error_invalid_number_of_cached_files:
            return out << "invalid number of cached files";
    }
    return out << "unknown";
}

std::string create_jit_report(const jit_info &info) {
    std::string report = fmt::format("{}; ", info.duration);
    if (info.use_ptx_inline) {
        report += "PTX inline; ";
    }
    report += fmt::format("cache: {} ({})", info.cache_state, info.cache_dir);
    return report;
}

}  // namespace plssvm::opencl::detail
