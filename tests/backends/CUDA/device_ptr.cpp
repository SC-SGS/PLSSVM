/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Tests for the CUDA backend device pointer.
*/

#include "plssvm/backends/CUDA/detail/device_ptr.cuh"

#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception

#include "../generic_device_ptr_tests.h"

#include "custom_test_macros.hpp"      // EXPECT_THROW_WHAT, EXPECT_FLOATING_POINT_VECTOR_NEAR
#include "naming.hpp"                  // naming::{real_type_kernel_function_to_name, real_type_to_name}
#include "types_to_test.hpp"           // util::{real_type_kernel_function_gtest, real_type_gtest}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::StartsWith
#include "gtest/gtest.h"           // TEST, EXPECT_GE, EXPECT_NO_THROW, ::testing::StartsWith

template <typename T>
class CUDADevicePtr : public ::testing::Test {};
TYPED_TEST_SUITE(CUDADevicePtr, util::real_type_gtest, naming::real_type_to_name); // TODO: other types also?

TYPED_TEST(CUDADevicePtr, default_construct) {
    generic::test_default_construct<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, construct) {
    generic::test_construct<plssvm::cuda::detail::device_ptr<TypeParam>>(0);
}
TYPED_TEST(CUDADevicePtr, move_construct) {
    generic::test_move_construct<plssvm::cuda::detail::device_ptr<TypeParam>>(0);
}
TYPED_TEST(CUDADevicePtr, move_assign) {
    generic::test_move_assign<plssvm::cuda::detail::device_ptr<TypeParam>>(0);
}

TYPED_TEST(CUDADevicePtr, swap) {
    generic::test_swap<plssvm::cuda::detail::device_ptr<TypeParam>>(0);
}

TYPED_TEST(CUDADevicePtr, memset) {
    generic::test_memset<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, memset_with_count) {
    generic::test_memset_with_count<plssvm::cuda::detail::device_ptr<TypeParam>>();
}

TYPED_TEST(CUDADevicePtr, fill) {
    generic::test_fill<plssvm::cuda::detail::device_ptr<TypeParam>>();
}

TYPED_TEST(CUDADevicePtr, copy_vector) {
    generic::test_copy_vector<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, copy_vector_exception) {
    generic::test_copy_vector_exception<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, copy_vector_with_count) {
    generic::test_copy_vector_with_count<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, copy_vector_with_count_exception) {
    generic::test_copy_vector_with_count_exception<plssvm::cuda::detail::device_ptr<TypeParam>>();
}

TYPED_TEST(CUDADevicePtr, copy_ptr) {
    generic::test_copy_ptr<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtr, copy_ptr_with_count) {
    generic::test_copy_ptr<plssvm::cuda::detail::device_ptr<TypeParam>>();
}


template <typename T>
class CUDADevicePtrDeathTest : public CUDADevicePtr<T> {};
TYPED_TEST_SUITE(CUDADevicePtrDeathTest, util::real_type_gtest, naming::real_type_to_name); // TODO: other types also?

TYPED_TEST(CUDADevicePtrDeathTest, memset) {
    generic::test_memset_death_test<plssvm::cuda::detail::device_ptr<TypeParam>>();
}

TYPED_TEST(CUDADevicePtrDeathTest, copy_ptr) {
    generic::test_copy_ptr_death_test<plssvm::cuda::detail::device_ptr<TypeParam>>();
}
TYPED_TEST(CUDADevicePtrDeathTest, copy_ptr_with_count) {
    generic::test_copy_ptr_with_count_death_test<plssvm::cuda::detail::device_ptr<TypeParam>>();
}

// TODO: add missing death tests

