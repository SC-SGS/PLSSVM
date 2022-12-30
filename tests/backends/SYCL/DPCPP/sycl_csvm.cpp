/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the SYCL backend using DPC++ as SYCL implementation.
 */

#include "backends/SYCL/DPCPP/mock_dpcpp_csvm.hpp"

#include "plssvm/backend_types.hpp"                         // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/SYCL/DPCPP/csvm.hpp"              // plssvm::dpcpp::csvm
#include "plssvm/backends/SYCL/exceptions.hpp"              // plssvm::dpcpp::backend_exception
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"                 // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                             // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include "../../../custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "../../../utility.hpp"             // util::redirect_output
#include "../../generic_csvm_tests.hpp"     // generic CSVM tests to instantiate

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, TYPED_TEST_SUITE, TYPED_TEST, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::{Test, Types}

#include <tuple>    // std::make_tuple, std::get
#include <utility>  // std::make_pair

class DPCPPCSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(DPCPPCSVM, construct_parameter) {
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::dpcpp::csvm{ plssvm::parameter{} });
}
TEST_F(DPCPPCSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, params }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, params }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, params }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, params }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(DPCPPCSVM, construct_target_and_named_args) {
    // every target is allowed for SYCL
#if defined(PLSSVM_HAS_CPU_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::cpu,
                                            plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                            plssvm::cost = 2.0,
                                            plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_nvidia,
                                            plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                            plssvm::cost = 2.0,
                                            plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_AMD_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    // EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::cost = 2.0 })); // TODO: @breyerml which wone would you like to test?
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_amd,
                                            plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                            plssvm::cost = 2.0,
                                            plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
#if defined(PLSSVM_HAS_INTEL_TARGET)
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel, plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }));
#else
    EXPECT_THROW_WHAT((plssvm::dpcpp::csvm{ plssvm::target_platform::gpu_intel,
                                            plssvm::kernel_type = plssvm::kernel_function_type::linear,
                                            plssvm::cost = 2.0,
                                            plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range }),
                      plssvm::dpcpp::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

template <typename T, plssvm::kernel_function_type kernel, plssvm::sycl::kernel_invocation_type invocation>
struct csvm_test_type {
    using mock_csvm_type = mock_dpcpp_csvm;
    using csvm_type = plssvm::dpcpp::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
    inline static auto additional_arguments = std::make_tuple(std::make_pair(plssvm::sycl_kernel_invocation_type, invocation));
};

class csvm_test_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}_{}_{}_{}",
                           plssvm::csvm_to_backend_type_v<typename T::csvm_type>,
                           plssvm::detail::arithmetic_type_name<typename T::real_type>(),
                           T::kernel_type,
                           std::get<0>(T::additional_arguments).second);
    }
};

using csvm_test_types = ::testing::Types<
    csvm_test_type<float, plssvm::kernel_function_type::linear, plssvm::sycl::kernel_invocation_type::nd_range>,
    csvm_test_type<float, plssvm::kernel_function_type::polynomial, plssvm::sycl::kernel_invocation_type::nd_range>,
    csvm_test_type<float, plssvm::kernel_function_type::rbf, plssvm::sycl::kernel_invocation_type::nd_range>,
    csvm_test_type<double, plssvm::kernel_function_type::linear, plssvm::sycl::kernel_invocation_type::nd_range>,
    csvm_test_type<double, plssvm::kernel_function_type::polynomial, plssvm::sycl::kernel_invocation_type::nd_range>,
    csvm_test_type<double, plssvm::kernel_function_type::rbf, plssvm::sycl::kernel_invocation_type::nd_range>,

    csvm_test_type<float, plssvm::kernel_function_type::linear, plssvm::sycl::kernel_invocation_type::hierarchical>,
    csvm_test_type<float, plssvm::kernel_function_type::polynomial, plssvm::sycl::kernel_invocation_type::hierarchical>,
    csvm_test_type<float, plssvm::kernel_function_type::rbf, plssvm::sycl::kernel_invocation_type::hierarchical>,
    csvm_test_type<double, plssvm::kernel_function_type::linear, plssvm::sycl::kernel_invocation_type::hierarchical>,
    csvm_test_type<double, plssvm::kernel_function_type::polynomial, plssvm::sycl::kernel_invocation_type::hierarchical>,
    csvm_test_type<double, plssvm::kernel_function_type::rbf, plssvm::sycl::kernel_invocation_type::hierarchical>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackend, GenericCSVM, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackendDeathTest, GenericCSVMDeathTest, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackend, GenericGPUCSVM, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(DPCPPBackendDeathTest, GenericGPUCSVMDeathTest, csvm_test_types, csvm_test_type_to_name);
