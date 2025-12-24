/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the stdpar backend using the NVHPC stdpar implementation.
 */

#include "plssvm/backends/stdpar/csvm.hpp"        // plssvm::stdpar::{csvc, csvr}
#include "plssvm/backends/stdpar/exceptions.hpp"  // plssvm::stdpar::backend_exception
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::parameter, plssvm::detail::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{cartesian_type_product_t, test_parameter_type_at_t}
#include "tests/utility.hpp"             // util::redirect_output

#include "fmt/format.h"   // NOLINT: fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_NO_THROW, ::testing::Test

#include <tuple>  // std::tuple

using stdpar_csvm_types_list = std::tuple<plssvm::stdpar::csvc, plssvm::stdpar::csvr>;
using stdpar_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<stdpar_csvm_types_list>>;

template <typename T>
class NVHPCStdparCSVMConstructor : public ::testing::Test,
                                   private util::redirect_output<> {
  protected:
    using fixture_csvm_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(NVHPCStdparCSVMConstructor, stdpar_csvm_types_gtest, naming::test_parameter_to_name);

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(NVHPCStdparCSVMConstructor, DefaultConstruct) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET) || defined(PLSSVM_HAS_NVIDIA_TARGET)
    // default constructor must always work
    EXPECT_NO_THROW(csvm_type{});
#else
    EXPECT_THROW_WHAT((csvm_type{}),
                      plssvm::stdpar::backend_exception,
                      fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", plssvm::determine_default_target_platform()));
#endif
}

TYPED_TEST(NVHPCStdparCSVMConstructor, ConstructParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET) || defined(PLSSVM_HAS_NVIDIA_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(csvm_type{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::parameter{} }),
                      plssvm::stdpar::backend_exception,
                      fmt::format("Requested target platform '{}' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!", plssvm::determine_default_target_platform()));
#endif
}

TYPED_TEST(NVHPCStdparCSVMConstructor, ConstructTargetAndParameter) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, params }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, params }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_amd' for the nvhpc stdpar backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_intel' for the nvhpc stdpar backend!");
}

TYPED_TEST(NVHPCStdparCSVMConstructor, ConstructNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET) || defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TYPED_TEST(NVHPCStdparCSVMConstructor, ConstructTargetAndNamedArgs) {
    using csvm_type = typename TestFixture::fixture_csvm_type;

#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_amd' for the nvhpc stdpar backend!");
    EXPECT_THROW_WHAT((csvm_type{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::stdpar::backend_exception,
                      "Invalid target platform 'gpu_intel' for the nvhpc stdpar backend!");
}
