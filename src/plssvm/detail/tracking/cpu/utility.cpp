/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/cpu/utility.hpp"

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::split_as
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::hardware_sampling_exception

#include "fmt/format.h"  // fmt::format
#include "subprocess.h"  // subprocess_s, subprocess_create, subprocess_join, subprocess_stdout, subprocess_stderr, subprocess_option_e

#include <algorithm>    // std::transform
#include <cstddef>      // std::size_t
#include <cstdio>       // std::FILE, std::fread
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail::tracking {

std::string run_subprocess(const std::string_view cmd_line) {
    // search PATH for executable
    constexpr int options = subprocess_option_e::subprocess_option_search_user_path | subprocess_option_e::subprocess_option_combined_stdout_stderr;
    constexpr static std::string::size_type buffer_size = 4096;

    // extract the separate command line arguments
    const std::vector<std::string> cmd_split = detail::split_as<std::string>(cmd_line, ' ');
    // convert to pointers
    std::vector<const char *> cmd_ptr_split(cmd_split.size());
    std::transform(cmd_split.cbegin(), cmd_split.cend(), cmd_ptr_split.begin(), [](const std::string &s) { return s.data(); });
    cmd_ptr_split.push_back(nullptr);  // subprocess wants the array to be terminated by a nullptr

    // create subprocess
    subprocess_s proc{};
    PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_create(cmd_ptr_split.data(), options, &proc));
    // wait until process has finished
    int return_code{};
    PLSSVM_SUBPROCESS_ERROR_CHECK(subprocess_join(&proc, &return_code));
    if (return_code != 0) {
        throw hardware_sampling_exception{ fmt::format("Error: \"{}\" returned with {}!", cmd_line, return_code) };
    }

    // get output handle and read data -> stdout and stderr are the same handle
    std::FILE *out_handle = subprocess_stdout(&proc);
    std::string buffer(buffer_size, '\0');  // 4096 characters should be enough
    const std::size_t bytes_read = std::fread(buffer.data(), sizeof(typename decltype(buffer)::value_type), buffer.size(), out_handle);

    // create output
    return buffer.substr(0, bytes_read);
}

}  // namespace plssvm::detail::tracking
