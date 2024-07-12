/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the stdpar backend using the GNU GCC + TBB stdpar implementation.
 */

#include "plssvm/backends/stdpar/csvm.hpp"        // plssvm::stdpar::csvm
#include "plssvm/backends/stdpar/exceptions.hpp"  // plssvm::stdpar::backend_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/utility.hpp"             // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, ::testing::Test

class gnu_tbb_stdparCSVM : public ::testing::Test,
                           private util::redirect_output<> { };

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(gnu_tbb_stdparCSVM, construct_parameter) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::stdpar::csvm{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::parameter{} }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TEST_F(gnu_tbb_stdparCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the stdpar backend using GNU GCC + TBB
    EXPECT_NO_THROW((plssvm::stdpar::csvm{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((plssvm::stdpar::csvm{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::automatic, params }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the gnu_tbb stdpar backend!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_amd' for the gnu_tbb stdpar backend!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_intel' for the gnu_tbb stdpar backend!");
}

TEST_F(gnu_tbb_stdparCSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_CPU_TARGET)
    // only automatic or cpu are allowed as target platform for the stdpar backend using GNU GCC + TBB
    EXPECT_NO_THROW((plssvm::stdpar::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::stdpar::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_nvidia' for the gnu_tbb stdpar backend!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_amd' for the gnu_tbb stdpar backend!");
    EXPECT_THROW_WHAT((plssvm::stdpar::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_intel' for the gnu_tbb stdpar backend!");
}
