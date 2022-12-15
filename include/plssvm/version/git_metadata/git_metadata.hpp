/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements compile-time constants to query git information.
 */

#ifndef PLSSVM_VERSION_GIT_METADATA_GIT_METADATA_HPP_
#define PLSSVM_VERSION_GIT_METADATA_GIT_METADATA_HPP_
#pragma once

#include <string_view>  // std::string_view

namespace plssvm::version::git_metadata {

/**
 * @brief Check whether the metadata has been successfully populated.
 * @details There may be no metadata of there wasn't a .git directory (e.g., downloaded source code without revision history).
 * @return `true` if the metadata has been successfully populated, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool is_populated();
/**
 * @brief Check whether there are any uncommitted changes that won't be reflected in the CommitID.
 * @return `true` if there are uncommitted changes, otherwise `false` (`[[nodiscard]]`)
 */
[[nodiscard]] bool has_uncommitted_changes();
/**
 * @brief The name of the author of this commit.
 * @return the configured name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view author_name();
/**
 * @brief The E-Mail of the author of this commit.
 * @return the configured E-Mail (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view author_email();
/**
 * @brief The full hash of the current commit.
 * @return the configured commit hash (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view commit_sha1();
/**
 * @brief The ISO 8601 commit date.
 * @return the configured commit date (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view commit_date();
/**
 * @brief The commit subject.
 * @return the configured subject (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view commit_subject();
/**
 * @brief The commit body.
 * @return the configured body (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view commit_body();
/**
 * @brief The commit description.
 * @return the configured description (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view describe();
/**
 * @brief The symbolic reference tied to HEAD.
 * @return the configured current branch name (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view branch();
/**
 * @brief The remote repository URL.
 * @return the configured remote URL (`[[nodiscard]]`)
 */
[[nodiscard]] std::string_view remote_url();

}  // namespace plssvm::version::git_metadata

#endif  // PLSSVM_VERSION_GIT_METADATA_GIT_METADATA_HPP_