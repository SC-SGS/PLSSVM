/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/version/version.hpp"                    // plssvm::version::{name, version}, plssvm::version::detail::{target_platform, target_platform}
#include "plssvm/backend_types.hpp"                      // plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::detail::list_available_sycl_implementations
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::{is_populated, commit_date, remote_url, branch, commit_sha1}

#include "fmt/format.h"   // fmt::format, fmt::join
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <optional>     // std::optional, std::make_optional, std::nullopt
#include <string>       // std::string
#include <string_view>  // std::string_view

namespace plssvm::version::detail {

std::optional<std::string> get_git_info() {
    if (git_metadata::is_populated()) {
        // git available -> get detailed information
        std::string_view date = git_metadata::commit_date();
        date.remove_suffix(date.size() - date.find_last_of(' '));
        return std::make_optional(fmt::format("({} {} {} ({}))", git_metadata::remote_url(), git_metadata::branch(), git_metadata::commit_sha1(), date));
    } else {
        // no git information available
        return std::nullopt;
    }
}

std::string get_version_info(const std::string_view executable_name, const bool with_backend_info) {
    // print specific information if requested
    std::string backend_specifics;
    if (with_backend_info) {
        backend_specifics += fmt::format("  PLSSVM_TARGET_PLATFORMS: {}\n", target_platforms);
        backend_specifics += fmt::format("  available target platforms: {}\n", fmt::join(list_available_target_platforms(), ", "));
        backend_specifics += fmt::format("  available backends: {}\n", fmt::join(list_available_backends(), ", "));
#if defined(PLSSVM_HAS_SYCL_BACKEND)
        backend_specifics += fmt::format("  available SYCL implementations: {}\n", fmt::join(::plssvm::sycl::list_available_sycl_implementations(), ", "));
#endif
    }

    return fmt::format(
        "{} v{} "
        "{}\n\n"
        "{}\n"
        "{}"
        "\n{}\n",
        executable_name,
        plssvm::version::version,
        get_git_info().value_or(""),
        plssvm::version::name,
        backend_specifics,
        copyright_notice);
}

}  // namespace plssvm::version::detail