/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/utility.hpp"

#include "plssvm/detail/assert.hpp"                        // PLSSVM_ASSERT
#include "plssvm/detail/data_distribution.hpp"             // plssvm::detail::data_distribution::maximum_local_memory_needed
#include "plssvm/detail/memory_size.hpp"                   // plssvm::detail::memory_size, custom memory size literals
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::kernel_launch_resources

#include "fmt/format.h"  // fmt::format

#if __has_include(<unistd.h>)
    #include <unistd.h>  // sysconf, _SC_PHYS_PAGES, _SC_PAGE_SIZE
    #define PLSSVM_UNIX_AVAILABLE_MEMORY
#elif __has_include(<windows.h>)
    #include <windows.h>  // MEMORYSTATUSEX, GlobalMemoryStatusEx
    #define PLSSVM_WINDOWS_AVAILABLE_MEMORY
#endif

#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    #include <cstddef>  // std::size_t
#endif

#include <cstdlib>   // std::getenv
#include <ctime>     // std::time_t, std::time, std:tm, std::localtime
#include <optional>  // std::optional, std::make_optional, std::nullopt
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail {

void check_local_memory_usage(const std::vector<std::optional<memory_size>> &local_memory) {
    PLSSVM_ASSERT(!local_memory.empty(), "At least one local memory value must be available since at least one place must always be present!");

    // determine the used local memory and check whether it exceeds the maximum necessary value!
    constexpr detail::memory_size required_local_memory_per_device = detail::data_distribution::maximum_local_memory_needed();
    for (const std::optional<detail::memory_size> &available_local_memory : local_memory) {
        if (available_local_memory.has_value() && required_local_memory_per_device > available_local_memory.value()) {
            // we need more local memory than available -> throw an exception
            throw kernel_launch_resources{ fmt::format("At least {} of local memory must be available, but available are only {}!",
                                                       required_local_memory_per_device,
                                                       available_local_memory.value()) };
        }
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "needed_local_memory", required_local_memory_per_device }));
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
    // post-process the local_memory vector for a better performance tracker output
    std::vector<memory_size> processed_local_memory(local_memory.size());
    for (std::size_t i = 0; i < processed_local_memory.size(); ++i) {
        processed_local_memory[i] = local_memory[i].value_or(0_B);
    }
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "available_local_memory_per_place", processed_local_memory }));
#endif
}

std::string current_date_time() {
    const std::time_t t = std::time(nullptr);
    const std::tm tm = *std::localtime(&t);
    return fmt::format("{:%Y-%m-%d %H:%M:%S}", tm);
}

memory_size get_system_memory() {
#if defined(PLSSVM_UNIX_AVAILABLE_MEMORY)
    const auto pages = static_cast<unsigned long long>(sysconf(_SC_PHYS_PAGES));  // vs. _SC_AVPHYS_PAGES
    const auto page_size = static_cast<unsigned long long>(sysconf(_SC_PAGE_SIZE));
    return memory_size{ pages * page_size };
#elif defined(PLSSVM_WINDOWS_AVAILABLE_MEMORY)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return memory_size{ status.ullTotalPhys };
#else
    return memory_size{ 0 };
#endif
}

std::optional<std::string> get_env_variable(const std::string &env_variable) {
    if (const char *env_value = std::getenv(env_variable.c_str()); env_value != nullptr) {
        return std::make_optional(env_value);
    }
    return std::nullopt;
}

#undef PLSSVM_UNIX_AVAILABLE_MEMORY
#undef PLSSVM_WINDOWS_AVAILABLE_MEMORY

}  // namespace plssvm::detail
