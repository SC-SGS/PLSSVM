/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Contains the googletest main function. Sets the DeathTest to "threadsafe" execution instead of "fast".
 */

#include "plssvm/backend_types.hpp"          // plssvm::backend_type
#include "plssvm/environment.hpp"            // plssvm::environment::scope_guard
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception

#include "hpx/hpx_init.hpp"  // hpx::init

#include "gtest/gtest.h"  // RUN_ALL_TESTS, ::testing::{InitGoogleTest, GTEST_FLAG},GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST definitions

#include <iostream>  // std::cerr, std::endl
#include <memory>    // std::unique_ptr, std::make_unique
#include <vector>    // std::vector

// silence GTest warnings/test errors

// generic C-SVM tests
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
// generic GPU C-SVM tests
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

// NOLINTBEGIN: hpx_main MUST be in global namespace, internal linkage DOES NOT work

[[nodiscard]] int hpx_main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
    // be sure that the HPX runtime is only started ONCE for all test invocations
    const int result = RUN_ALL_TESTS();
    ::hpx::finalize();
    return result;
}

// NOLINTEND

int main(int argc, char **argv) {
    // may throw an exception if the required level of MPI parallelism isn't available (really rare)
    std::unique_ptr<plssvm::environment::scope_guard> mpi_guard{};
    try {
        // initialize MPI environment via the plssvm::scope_guard
        mpi_guard = std::make_unique<plssvm::environment::scope_guard>(std::vector<plssvm::backend_type>{});
    } catch (const plssvm::mpi_exception &e) {
        std::cerr << "An exception occurred while setting up the MPI environment!: " << e.what_with_loc() << std::endl;
    }

    ::testing::InitGoogleTest(&argc, argv);

    // prevent problems with fork() in the presence of multiple threads
    // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
    // NOTE: may reduce performance of the (death) tests
#if !defined(_WIN32)
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
#endif
    return ::hpx::init(argc, argv);
}
