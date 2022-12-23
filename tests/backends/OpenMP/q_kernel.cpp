/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions generating the q vector using the OpenMP backend.
 */

#include "plssvm/backends/OpenMP/q_kernel.hpp"

#include "../../naming.hpp"         // naming::real_type_to_name
#include "../../types_to_test.hpp"  // util::real_type_gtest

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_DEATH, ::testing::Test

#include <vector>  // std::vector

template <typename T>
class OpenMPQKernelDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMPQKernelDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(OpenMPQKernelDeathTest, linear) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    std::vector<real_type> q(1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_q_linear(q, data), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));
}
TYPED_TEST(OpenMPQKernelDeathTest, polynomial) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    std::vector<real_type> q(1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_q_polynomial(q, data, 2, real_type{ 0.1 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));

    q.resize(data.size() - 1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_q_polynomial(q, data, 2, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}
TYPED_TEST(OpenMPQKernelDeathTest, rbf) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    std::vector<real_type> q(1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_q_rbf(q, data, real_type{ 0.1 }), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));

    q.resize(data.size() - 1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_q_rbf(q, data, real_type{ 0.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}