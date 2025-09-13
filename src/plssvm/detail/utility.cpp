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
#include "plssvm/detail/memory_size.hpp"                   // plssvm::detail::memory_size
#include "plssvm/detail/tracking/performance_tracker.hpp"  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY, plssvm::detail::tracking::tracking_entry
#include "plssvm/exceptions/exceptions.hpp"                // plssvm::kernel_launch_resources

#include "fmt/chrono.h"  // fmt::localtime
#include "fmt/format.h"  // fmt::format

#if __has_include(<unistd.h>)
    #include <unistd.h>  // sysconf, _SC_PHYS_PAGES, _SC_PAGE_SIZE
    #define PLSSVM_UNIX_AVAILABLE_MEMORY
#elif __has_include(<windows.h>)
    #include <windows.h>  // MEMORYSTATUSEX, GlobalMemoryStatusEx
    #define PLSSVM_WINDOWS_AVAILABLE_MEMORY
#endif

#include <ctime>     // std::time
#include <optional>  // std::optional
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
    PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((detail::tracking::tracking_entry{ "resource_constraints", "available_local_memory_per_place", local_memory }));
}

std::string current_date_time() {
    return fmt::format("{:%Y-%m-%d %H:%M:%S}", fmt::localtime(std::time(nullptr)));
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

#undef PLSSVM_UNIX_AVAILABLE_MEMORY
#undef PLSSVM_WINDOWS_AVAILABLE_MEMORY

}  // namespace plssvm::detail
