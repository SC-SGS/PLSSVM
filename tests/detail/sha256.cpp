/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the sha256 implementation.
 */

#include "plssvm/detail/sha256.hpp"                // plssvm::detail::sha256

#include "gtest/gtest.h"  // TEST, ASSERT_EQ, EXPECT_EQ

#include <string>  // std::string
#include <vector>  // std::vector


TEST(Base_Detail, sha256) {
    // https://www.di-mgt.com.au/sha_testvectors.html
    // example SHA256 input strings
    const std::vector<std::string> inputs = {
        "abc",
        "",
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu",
        std::string(1000000, 'a'),
        "test",
        "#include <iostream>\n"
        "int main() {\n"
        "  std::cout << \"Hello, World\" << std::endl;\n"
        "  return 0;\n"
        "}"
    };
    // corresponding SHA256 output strings
    const std::vector<std::string> correct_outputs = {
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1",
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
        "3a88d6a9c65fdb1a2c742bca7d5160728fd7da74d9f319527bc8d6c1e1739de0"
    };

    // check if the number of input and output matches
    ASSERT_EQ(inputs.size(), correct_outputs.size());

    // create sha265 hasher instance and check for correct outputs
    plssvm::detail::sha256 hasher{};
    for (std::vector<std::string>::size_type i = 0; i < inputs.size(); ++i) {
        // calculate sha256 of the input string
        const std::string sha = hasher(inputs[i]);
        // check sha256 output size (in hex format)
        EXPECT_EQ(sha.size(), 64);
        // check sha256 string for correctness
        EXPECT_EQ(sha, correct_outputs[i]) << "input: " << inputs[i] << ", output: " << sha << ", correct output: " << correct_outputs[i];
    }
}