/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions performing the actual kernel calculations using the OpenMP backend.
 */

#include "plssvm/backends/OpenMP/svm_kernel.hpp"

#include "../../naming.hpp"         // naming::real_type_to_name
#include "../../types_to_test.hpp"  // util::real_type_gtest

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_DEATH, ::testing::Test

#include <vector>  // std::vector

template <typename T>
class OpenMPSVMKernelDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMPSVMKernelDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(OpenMPSVMKernelDeathTest, polynomial) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> q(data.size() - 1);
    std::vector<real_type> ret(data.size() - 1);
    const std::vector<real_type> d(data.size() - 1);
    const real_type QA_cost{};
    const real_type cost{ 1.0 };
    const real_type add{ 1.0 };
    EXPECT_DEATH(plssvm::openmp::device_kernel_polynomial(q, ret, d, data, QA_cost, cost, add, 2, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}
TYPED_TEST(OpenMPSVMKernelDeathTest, rbf) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> q(data.size() - 1);
    std::vector<real_type> ret(data.size() - 1);
    const std::vector<real_type> d(data.size() - 1);

    EXPECT_DEATH(plssvm::openmp::device_kernel_rbf(q, ret, d, data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 1.0 }, real_type{ 0.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}

TYPED_TEST(OpenMPSVMKernelDeathTest, device_kernel) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> correct_data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> correct_q(correct_data.size() - 1);
    std::vector<real_type> correct_ret(correct_data.size() - 1);
    const std::vector<real_type> correct_d(correct_data.size() - 1);

    EXPECT_DEATH(plssvm::openmp::device_kernel_linear(std::vector<real_type>(1), correct_ret, correct_d, correct_data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));
    std::vector<real_type> ret(1);
    EXPECT_DEATH(plssvm::openmp::device_kernel_linear(correct_q, ret, correct_d, correct_data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 2 != 1"));
    EXPECT_DEATH(plssvm::openmp::device_kernel_linear(correct_q, correct_ret, std::vector<real_type>(1), correct_data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 2 != 1"));

    EXPECT_DEATH(plssvm::openmp::device_kernel_linear(correct_q, correct_ret, correct_d, correct_data, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("cost must not be 0.0 since it is 1 / plssvm::cost!"));
    EXPECT_DEATH(plssvm::openmp::device_kernel_linear(correct_q, correct_ret, correct_d, correct_data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 }), ::testing::HasSubstr("add must either be -1.0 or 1.0, but is 0!"));
}