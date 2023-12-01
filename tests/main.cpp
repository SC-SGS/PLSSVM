/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Contains the googletest main function. Sets the DeathTest to "threadsafe" execution instead of "fast".
 */

#include "gtest/gtest.h"  // GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST, RUN_ALL_TESTS, ::testing::{InitGoogleTest, GTEST_FLAG}

// silence GTest warnings/test errors
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVM);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolver);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunctionClassification);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverKernelFunctionClassification);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunctionDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericGPUCSVM);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericGPUCSVMKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtr);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtrLayout);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtrDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Exception);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // prevent problems with fork() in the presence of multiple threads
    // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
    // NOTE: may reduce performance of the (death) tests
#if !defined(_WIN32)
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
    return RUN_ALL_TESTS();
}