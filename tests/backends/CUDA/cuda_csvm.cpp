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

#include "plssvm/backends/CUDA/csvm.hpp"        // plssvm::cuda::csvm
#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception
#include "plssvm/detail/type_list.hpp"          // plssvm::detail::label_type_list
#include "plssvm/kernel_function_types.hpp"     // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/target_platforms.hpp"          // plssvm::target_platform

#include "backends/generic_csvm_tests.hpp"      // generic CSVM tests to instantiate
#include "backends/generic_gpu_csvm_tests.hpp"  // generic GPU CSVM tests to instantiate
#include "custom_test_macros.hpp"               // EXPECT_THROW_WHAT
#include "naming.hpp"                           // naming::test_parameter_to_name
#include "types_to_test.hpp"                    // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "utility.hpp"                          // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <tuple>  // std::make_tuple, std::tuple

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
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::automatic, params }),
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, params }),
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
TEST_F(CUDACSVM, construct_named_args) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the CUDA backend
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}
TEST_F(CUDACSVM, construct_target_and_named_args) {
#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    // only automatic or gpu_nvidia are allowed as target platform for the CUDA backend
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    EXPECT_NO_THROW((plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }));
#else
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::cuda::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    EXPECT_THROW_WHAT((plssvm::cuda::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::cost = 2.0 }),
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

struct cuda_csvm_test_type {
    using mock_csvm_type = mock_cuda_csvm;
    using csvm_type = plssvm::cuda::csvm;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static constexpr auto additional_arguments = std::make_tuple();
};
using cuda_csvm_test_tuple = std::tuple<cuda_csvm_test_type>;
using cuda_csvm_test_label_type_list = util::cartesian_type_product_t<cuda_csvm_test_tuple, plssvm::detail::supported_label_types>;
using cuda_csvm_test_type_list = util::cartesian_type_product_t<cuda_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using cuda_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_type_list>;
using cuda_solver_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_type_list, util::solver_type_list>;
using cuda_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_type_list, util::kernel_function_type_list>;
using cuda_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using cuda_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using cuda_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<cuda_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVM, cuda_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVMSolver, cuda_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVMKernelFunction, cuda_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVMSolverKernelFunction, cuda_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVMKernelFunctionClassification, cuda_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericCSVMSolverKernelFunctionClassification, cuda_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVMDeathTest, GenericCSVMSolverDeathTest, cuda_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVMDeathTest, GenericCSVMKernelFunctionDeathTest, cuda_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericGPUCSVM, cuda_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(CUDACSVM, GenericGPUCSVMKernelFunction, cuda_kernel_function_type_gtest, naming::test_parameter_to_name);
