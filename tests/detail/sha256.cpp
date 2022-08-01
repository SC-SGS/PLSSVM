/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the sha256 implementation.
 */

#include "plssvm/detail/sha256.hpp"  // plssvm::detail::sha256

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ

#include <string>  // std::string
#include <vector>  // std::vector

using sha256_param_type = std::pair<std::string, std::string_view>;

class Sha256 : public ::testing::TestWithParam<sha256_param_type> {};

TEST_P(Sha256, correct_encoding) {
    // get generated parameter
    const sha256_param_type params = GetParam();

    // create sha265 hasher instance and check for correct outputs
    const plssvm::detail::sha256 hasher{};
    // calculate sha256 of the input string
    const std::string sha = hasher(params.first);
    // check sha256 output size (in hex format)
    ASSERT_EQ(sha.size(), 64);
    // check sha256 string for correctness
    EXPECT_EQ(sha, params.second) << fmt::format("input: {}, output: {}, correct output: {}", params.first, sha, params.second);
}

std::vector<sha256_param_type> sha256_test_values = {
    { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    { "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu", "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
    { std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0" },
    { "test", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" },
    { "#include <iostream>\nint main() {\n  std::cout << \"Hello, World\" << std::endl;\n  return 0;\n}", "3a88d6a9c65fdb1a2c742bca7d5160728fd7da74d9f319527bc8d6c1e1739de0" }
};

INSTANTIATE_TEST_SUITE_P(Sha256, Sha256, ::testing::ValuesIn(sha256_test_values));