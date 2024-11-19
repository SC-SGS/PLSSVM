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

#include "plssvm/environment.hpp"  // plssvm::environment::scope_guard

#include "gtest/gtest.h"  // RUN_ALL_TESTS, ::testing::{InitGoogleTest, GTEST_FLAG},GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST definitions

#include <cstdlib>  // std::atexit

// silence GTest warnings/test errors

// generic CSVM tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVM);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolver);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunctionClassification);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverKernelFunctionClassification);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMKernelFunctionDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericCSVMSolverKernelFunctionDeathTest);
// generic GPU CSVM tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericGPUCSVM);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericGPUCSVMKernelFunction);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(GenericGPUCSVMDeathTest);
// pinned memory tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PinnedMemory);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PinnedMemoryLayout);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PinnedMemoryDeathTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(PinnedMemoryLayoutDeathTest);
// device pointer tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtr);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtrLayout);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DevicePtrDeathTest);
// exception tests
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Exception);

void ensure_finalization() {
    plssvm::environment::finalize();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // initialize environments
    const plssvm::environment::scope_guard environment_guard{};
    // Note: necessary for Kokkos::SYCL
    [[maybe_unused]] const int ret = std::atexit(ensure_finalization);

    // prevent problems with fork() in the presence of multiple threads
    // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
    // NOTE: may reduce performance of the (death) tests
#if !defined(_WIN32)
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
    return RUN_ALL_TESTS();
}
