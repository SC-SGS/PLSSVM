/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/utility.hpp"

#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size

#include "fmt/chrono.h"  // fmt::localtime
#include "fmt/core.h"    // fmt::format

#if __has_include(<unistd.h>)
    #include <unistd.h>  // sysconf, _SC_PHYS_PAGES, _SC_PAGE_SIZE
    #define PLSSVM_UNIX_AVAILABLE_MEMORY
#elif __has_include(<windows.h>)
    #include <windows.h>  //
    #define PLSSVM_WINDOWS_AVAILABLE_MEMORY
#endif

#include <ctime>   // std::time
#include <string>  // std::string

namespace plssvm::detail {

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