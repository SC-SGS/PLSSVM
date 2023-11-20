/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the Git metadata queries.
 */

#include "plssvm/version/git_metadata/git_metadata.hpp"

#include "gmock/gmock-matchers.h"  // ::testing::{HasSubstr, Not, StartsWith, EndsWith}
#include "gtest/gtest.h"           // TEST, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT

#include <regex>   // std::regex, std::regex::extended, std::regex_match
#include <string>  // std::string

using namespace plssvm::version;

TEST(GitMetadata, author_name) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the author name must not be empty
        EXPECT_FALSE(git_metadata::author_name().empty());
    } else {
        // if we are outside a Git repository, the author name must be empty
        EXPECT_TRUE(git_metadata::author_name().empty());
    }
}

TEST(GitMetadata, author_email) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the author email must not be empty
        EXPECT_FALSE(git_metadata::author_email().empty());
        // check for a valid email address
        EXPECT_TRUE(std::regex_match(std::string{ git_metadata::author_email() },
                                     std::regex{ "^[[:graph:]]+@[[:graph:]]+\\.[[:graph:]]+$", std::regex::extended }));
    } else {
        // if we are outside a Git repository, the author email must be empty
        EXPECT_TRUE(git_metadata::author_email().empty());
    }
}

TEST(GitMetadata, commit_sha1) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the commit sha1 must not be empty
        EXPECT_FALSE(git_metadata::commit_sha1().empty());
        // test for valid commit sha1 characters
        EXPECT_TRUE(std::regex_match(std::string{ git_metadata::commit_sha1() },
                                     std::regex{ "^([0-9a-f]{40})|([0-9a-f]{6,8})$", std::regex::extended }));
    } else {
        // if we are outside a Git repository, the commit sha1 must be empty
        EXPECT_TRUE(git_metadata::commit_sha1().empty());
    }
}

TEST(GitMetadata, commit_date) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the commit date must not be empty
        EXPECT_FALSE(git_metadata::commit_date().empty());
    } else {
        // if we are outside a Git repository, the commit date must be empty
        EXPECT_TRUE(git_metadata::commit_date().empty());
    }
}

TEST(GitMetadata, commit_subject) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the commit subject must not be empty
        EXPECT_FALSE(git_metadata::commit_subject().empty());
    } else {
        // if we are outside a Git repository, the commit subject must be empty
        EXPECT_TRUE(git_metadata::commit_subject().empty());
    }
}

TEST(GitMetadata, commit_body) {
    if (!git_metadata::is_populated()) {
        // if we are outside a Git repository, the commit body must be empty
        EXPECT_TRUE(git_metadata::commit_subject().empty());
    }
}

TEST(GitMetadata, describe) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the description message must not be empty
        EXPECT_FALSE(git_metadata::describe().empty());
    } else {
        // if we are outside a Git repository, the description message must be empty
        EXPECT_TRUE(git_metadata::describe().empty());
    }
}

TEST(GitMetadata, branch) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the branch must not be empty
        EXPECT_FALSE(git_metadata::branch().empty());
        // check whether the branch name is valid
        // https://git-scm.com/docs/git-check-ref-format
        EXPECT_THAT(std::string{ git_metadata::branch() }, ::testing::Not(::testing::StartsWith(".")));    // must not start with a .a
        EXPECT_THAT(std::string{ git_metadata::branch() }, ::testing::Not(::testing::EndsWith(".lock")));  // must not end with .lock
        EXPECT_THAT(std::string{ git_metadata::branch() }, ::testing::Not(::testing::EndsWith("..")));     // must not end with .. anywhere
        // EXPECT_THAT(git_metadata::branch(), ::testing::Not(::testing::ContainsRegex("[\\\040-\\\177]+")));  // not contain an ASCII control character anywhere
        EXPECT_FALSE(std::regex_match(std::string{ git_metadata::branch() },
                                      std::regex{ "(~|\\^|:)+", std::regex::extended }));  // must not contain a ~, ^ or : anywhere
        EXPECT_FALSE(std::regex_match(std::string{ git_metadata::branch() },
                                      std::regex{ "(\\?|\\[|\\*)+", std::regex::extended }));            // must not contain ?, [ or * anywhere
        EXPECT_THAT(std::string{ git_metadata::branch() }, ::testing::Not(::testing::StartsWith("/")));  // must not start with a /
        EXPECT_THAT(std::string{ git_metadata::branch() }, ::testing::Not(::testing::EndsWith("/")));    // must not end with a /
        EXPECT_FALSE(std::regex_match(std::string{ git_metadata::branch() },
                                      std::regex{ "/{2,}", std::regex::extended }));  // must not contain multiple consecutive /
        EXPECT_FALSE(std::regex_match(std::string{ git_metadata::branch() },
                                      std::regex{ "(@\\{)+", std::regex::extended }));  // must not contain @{
        EXPECT_FALSE(std::regex_match(std::string{ git_metadata::branch() },
                                      std::regex{ "(@|\\\\)+", std::regex::extended }));  // must not contain a single @ or backslash
    } else {
        // if we are outside a Git repository, the branch name is HEAD
        EXPECT_EQ(git_metadata::branch(), "HEAD");
    }
}

TEST(GitMetadata, remote_url) {
    if (git_metadata::is_populated()) {
        // if we are inside a Git repository, the remote URL must not be empty
        EXPECT_FALSE(git_metadata::remote_url().empty());
        // check whether the remote URL is valid
        // https://github.com/jonschlinkert/is-git-url
        EXPECT_TRUE(std::regex_match(std::string{ git_metadata::remote_url() },
                                     std::regex{ "^(git|ssh|http|https|git@[A-Za-z0-9_\\.\\-]+):(//)?[A-Za-z0-9_\\.@:/-~\\-]+(\\.git(/)?)?$", std::regex::extended }));
    } else {
        // if we are outside a Git repository, the remote URL must be empty
        EXPECT_TRUE(git_metadata::remote_url().empty());
    }
}