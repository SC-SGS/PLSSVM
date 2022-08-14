/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions in the utility header.
 */

#include "plssvm/backend_types.hpp"

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains
#include "utility.hpp"                // util::{convert_to_string, convert_from_string}

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <sstream>  // std::istringstream
#include <vector>   // std::vector

// check whether the plssvm::backend_type -> std::string conversions are correct
TEST(BackendType, to_string) {
    // check conversions to std::string
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::automatic), "automatic");
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::openmp), "openmp");
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::cuda), "cuda");
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::hip), "hip");
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::opencl), "opencl");
    EXPECT_EQ(util::convert_to_string(plssvm::backend_type::sycl), "sycl");
}
TEST(Backend_Type, to_string_unknown) {
    // check conversions to std::string from unknown backend_type
    EXPECT_EQ(util::convert_to_string(static_cast<plssvm::backend_type>(6)), "unknown");
}

// check whether the std::string -> plssvm::backend_type conversions are correct
TEST(BackendType, from_string) {
    // check conversion from std::string
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("automatic"), plssvm::backend_type::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("AUTOmatic"), plssvm::backend_type::automatic);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("openmp"), plssvm::backend_type::openmp);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("OpenMP"), plssvm::backend_type::openmp);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("cuda"), plssvm::backend_type::cuda);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("CUDA"), plssvm::backend_type::cuda);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("hip"), plssvm::backend_type::hip);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("HIP"), plssvm::backend_type::hip);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("opencl"), plssvm::backend_type::opencl);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("OpenCL"), plssvm::backend_type::opencl);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("sycl"), plssvm::backend_type::sycl);
    EXPECT_EQ(util::convert_from_string<plssvm::backend_type>("SYCL"), plssvm::backend_type::sycl);
}
TEST(BackendType, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream ss{ "foo" };
    plssvm::backend_type b;
    ss >> b;
    EXPECT_TRUE(ss.fail());
}

TEST(BackendType, minimal_available_backend) {
    const std::vector<plssvm::backend_type> backends = plssvm::list_available_backends();

    // at least two backends must be available!
    EXPECT_GE(backends.size(), 2);

    // the automatic backend must always be present
    EXPECT_TRUE(plssvm::detail::contains(backends, plssvm::backend_type::automatic));
}