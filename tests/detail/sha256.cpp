/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the sha256 implementation.
 */

#include "plssvm/detail/sha256.hpp"

#include "naming.hpp"  // naming::pretty_print_sha256

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TEST_P, INSTANTIATE_TEST_SUITE_P, ASSERT_EQ, EXPECT_EQ, ::testing::{TestWithParam, Values}

#include <string>   // std::string
#include <utility>  // std::pair, std::make_pair
#include <vector>   // std::vector

class Sha256 : public ::testing::TestWithParam<std::pair<std::string, std::string_view>> {};

TEST_P(Sha256, correct_encoding) {
    // get generated parameter
    const auto [input, encoded_output] = GetParam();

    // create sha265 hasher instance and check for correct outputs
    const plssvm::detail::sha256 hasher{};
    // calculate sha256 of the input string
    const std::string sha = hasher(input);
    // check sha256 output size (in hex format)
    ASSERT_EQ(sha.size(), 64);
    // check sha256 string for correctness
    EXPECT_EQ(sha, encoded_output) << fmt::format(R"(input: "{}", output: {}, correct output: {})", input, sha, encoded_output);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(Sha256, Sha256, ::testing::Values(
        std::make_pair("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        std::make_pair("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        std::make_pair("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        std::make_pair("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu", "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
        std::make_pair(std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"),
        std::make_pair("test", "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"),
        std::make_pair("#include <iostream>\nint main() {\n  std::cout << \"Hello, World\" << std::endl;\n  return 0;\n}", "3a88d6a9c65fdb1a2c742bca7d5160728fd7da74d9f319527bc8d6c1e1739de0")),
        naming::pretty_print_sha256<Sha256>);
// clang-format on