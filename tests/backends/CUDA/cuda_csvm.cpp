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

#include "plssvm/backends/CUDA/csvm.hpp"        // plssvm::openmp::csvm
#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::openmp::backend_exception
#include "plssvm/data_set.hpp"                    // plssvm::data_set
#include "plssvm/kernel_function_types.hpp"       // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                   // plssvm::detail::data_set
#include "plssvm/target_platforms.hpp"            // plssvm::target_platform

#include "backends/compare.hpp"        // compare::{generate_q, calculate_w, kernel_function, device_kernel_function}
#include "backends/generic_tests.hpp"  // generic::{test_solve_system_of_linear_equations, test_predict_values, test_predict, test_score}
#include "custom_test_macros.hpp"      // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "naming.hpp"                  // naming::{real_type_kernel_function_to_name, real_type_to_name}
#include "types_to_test.hpp"           // util::{real_type_kernel_function_gtest, real_type_gtest}
#include "utility.hpp"                 // util::{redirect_output, generate_random_vector}

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, TYPED_TEST_SUITE, TYPED_TEST, ::testing::Test

#include <vector>  // std::vector

class CUDACSVM : public ::testing::Test, private util::redirect_output {};

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

TEST_F(CUDACSVM, num_available_devices) {
    generic::test_num_available_devices<mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMSolveSystemOfLinearEquations : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMSolveSystemOfLinearEquations, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMSolveSystemOfLinearEquations, solve_system_of_linear_equations_trivial) {
   SCOPED_TRACE("plssvm::kernel_function_type::linear");
   generic::test_solve_system_of_linear_equations<TypeParam, mock_cuda_csvm>(plssvm::kernel_function_type::linear);
   SCOPED_TRACE("plssvm::kernel_function_type::polynomial");
   generic::test_solve_system_of_linear_equations<TypeParam, mock_cuda_csvm>(plssvm::kernel_function_type::polynomial);
   // no tests for RBF since it is non-trivial to find a parameter set with a trivial solution
}

template <typename T>
class CUDACSVMPredictValues : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMPredictValues, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMPredictValues, predict_values) {
   SCOPED_TRACE("plssvm::kernel_function_type::linear");
   generic::test_predict_values<TypeParam, mock_cuda_csvm>(plssvm::kernel_function_type::linear);
   SCOPED_TRACE("plssvm::kernel_function_type::polynomial");
   generic::test_predict_values<TypeParam, mock_cuda_csvm>(plssvm::kernel_function_type::polynomial);
   // no tests for RBF since it is non-trivial to find a parameter set with a trivial solution
}

template <typename T>
class CUDACSVMGenerateQ : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMGenerateQ, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMGenerateQ, generate_q) {
   generic::test_generate_q<typename TypeParam::real_type, mock_cuda_csvm>(TypeParam::kernel_type);
}

template <typename T>
class CUDACSVMGenerateQDeathTest : public CUDACSVMGenerateQ<T> {};
TYPED_TEST_SUITE(CUDACSVMGenerateQDeathTest, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMGenerateQDeathTest, generate_q) {
   generic::test_generate_q_death_test<typename TypeParam::real_type, mock_cuda_csvm>(TypeParam::kernel_type);
}

template <typename T>
class CUDACSVMCalculateW : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMCalculateW, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMCalculateW, calculate_w) {
   generic::test_calculate_w<TypeParam, mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMCalculateWDeathTest : public CUDACSVMCalculateW<T> {};
TYPED_TEST_SUITE(CUDACSVMCalculateWDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMCalculateWDeathTest, calculate_w) {
   generic::test_calculate_w_death_test<TypeParam, mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMRunDeviceKernel : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMRunDeviceKernel, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMRunDeviceKernel, run_device_kernel) {
   generic::test_run_device_kernel<typename TypeParam::real_type, mock_cuda_csvm>(TypeParam::kernel_type);
}

template <typename T>
class CUDACSVMRunDeviceKernelDeathTest : public CUDACSVMRunDeviceKernel<T> {};
TYPED_TEST_SUITE(CUDACSVMRunDeviceKernelDeathTest, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMRunDeviceKernelDeathTest, run_device_kernel) {
   generic::test_run_device_kernel_death_test<typename TypeParam::real_type, mock_cuda_csvm>(TypeParam::kernel_type);
}

template <typename T>
class CUDACSVMDeviceReduction : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMDeviceReduction, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMDeviceReduction, device_reduction) {
   generic::test_device_reduction<TypeParam, mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMDeviceReductionDeathTest : public CUDACSVMDeviceReduction<T> {};
TYPED_TEST_SUITE(CUDACSVMDeviceReductionDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMDeviceReductionDeathTest, device_reduction) {
   generic::test_device_reduction_death_test<TypeParam, mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMSelectNumUsedDevices : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMSelectNumUsedDevices, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMSelectNumUsedDevices, select_num_used_devices) {
   generic::test_select_num_used_devices<typename TypeParam::real_type, mock_cuda_csvm>(TypeParam::kernel_type);
}

template <typename T>
class CUDACSVMSetupDataOnDevice : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMSetupDataOnDevice, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMSetupDataOnDevice, setup_data_on_device_minimal) {
   generic::test_setup_data_on_device_minimal<TypeParam, mock_cuda_csvm>();
}
TYPED_TEST(CUDACSVMSetupDataOnDevice, setup_data_on_device) {
   generic::test_setup_data_on_device<TypeParam, mock_cuda_csvm>();
}

template <typename T>
class CUDACSVMSetupDataOnDeviceDeathTest : public CUDACSVMSetupDataOnDevice<T> {};
TYPED_TEST_SUITE(CUDACSVMSetupDataOnDeviceDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(CUDACSVMSetupDataOnDeviceDeathTest, sanity_checks) {
    generic::test_setup_data_on_device_death_test<TypeParam, mock_cuda_csvm>();
}


template <typename T>
class CUDACSVMPredictAndScore : public CUDACSVM {};
TYPED_TEST_SUITE(CUDACSVMPredictAndScore, util::real_type_kernel_function_gtest, naming::real_type_kernel_function_to_name);

TYPED_TEST(CUDACSVMPredictAndScore, predict) {
   generic::test_predict<typename TypeParam::real_type, plssvm::cuda::csvm>(TypeParam::kernel_type);
}

TYPED_TEST(CUDACSVMPredictAndScore, score) {
   generic::test_score<typename TypeParam::real_type, plssvm::cuda::csvm>(TypeParam::kernel_type);
}