/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the CUDA backend.
 */

#include "backends/CUDA/mock_cuda_csvm.hpp"

#include "plssvm/backend_types.hpp"                // plssvm::csvm_to_backend_type_v
#include "plssvm/backends/CUDA/csvm.hpp"           // plssvm::cuda::csvm
#include "plssvm/backends/CUDA/exceptions.hpp"     // plssvm::cuda::backend_exception
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_function_types.hpp"        // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter, plssvm::kernel_type, plssvm::cost
#include "plssvm/target_platforms.hpp"             // plssvm::target_platform

#include "../../custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "../../utility.hpp"             // util::redirect_output
#include "../generic_csvm_tests.hpp"     // generic CSVM tests to instantiate

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, TYPED_TEST_SUITE, TYPED_TEST, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::{Test, Types}

#include <tuple>  // std::make_tuple

class CUDACSVM : public ::testing::Test, private util::redirect_output<> {};

// check whether the constructor correctly fails when using an incompatible target platform
TEST_F(CUDACSVM, construct_parameter) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // the automatic target platform must always be available
    EXPECT_NO_THROW(plssvm::cuda::csvm{ plssvm::parameter{} });
#else
    EXPECT_THROW_WHAT(plssvm::cuda::csvm{ plssvm::parameter{} },
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(CUDACSVM, construct_target_and_parameter) {
    // create parameter struct
    const plssvm::parameter params{};

#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the CUDA backend
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::automatic, params }));
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, params }));
#else
    EXPECT_THROW_WHAT(plssvm::cuda::csvm{ plssvm::target_platform::automatic, params },
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT(plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, params },
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'cpu' for the CUDA backend!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'gpu_amd' for the CUDA backend!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'gpu_intel' for the CUDA backend!");
}
TEST_F(CUDACSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the CUDA backend
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT(plssvm::cuda::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 },
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT(plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 },
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

    // all other target platforms must throw
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::cpu, plssvm::cost = 2.0 }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'cpu' for the CUDA backend!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_amd, plssvm::cost = 2.0 }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'gpu_amd' for the CUDA backend!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_intel, plssvm::cost = 2.0 }),
                      plssvm::cuda::backend_exception,
                      "Invalid target platform 'gpu_intel' for the CUDA backend!");
}

template <typename T, plssvm::kernel_function_type kernel>
struct csvm_test_type {
    using mock_csvm_type = mock_cuda_csvm;
    using csvm_type = plssvm::cuda::csvm;
    using real_type = T;
    static constexpr plssvm::kernel_function_type kernel_type = kernel;
    inline static auto additional_arguments = std::make_tuple();
};

class csvm_test_type_to_name {
  public:
    template <typename T>
    static std::string GetName(int) {
        return fmt::format("{}_{}_{}",
                           plssvm::csvm_to_backend_type_v<typename T::csvm_type>,
                           plssvm::detail::arithmetic_type_name<typename T::real_type>(),
                           T::kernel_type);
    }
};

using csvm_test_types = ::testing::Types<
    csvm_test_type<float, plssvm::kernel_function_type::linear>,
    csvm_test_type<float, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<float, plssvm::kernel_function_type::rbf>,
    csvm_test_type<double, plssvm::kernel_function_type::linear>,
    csvm_test_type<double, plssvm::kernel_function_type::polynomial>,
    csvm_test_type<double, plssvm::kernel_function_type::rbf>>;

// instantiate type-parameterized tests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDABackend, GenericCSVM, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDABackendDeathTest, GenericCSVMDeathTest, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDABackend, GenericGPUCSVM, csvm_test_types, csvm_test_type_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDABackendDeathTest, GenericGPUCSVMDeathTest, csvm_test_types, csvm_test_type_to_name);
