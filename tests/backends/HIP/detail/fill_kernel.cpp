/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the utility function to fill a array with a specific value.
 */

#include "plssvm/backends/HIP/detail/fill_kernel.hip.hpp"

#include "naming.hpp"         // util::test_parameter_to_name
#include "types_to_test.hpp"  // util::{real_type_gtest, test_parameter_type_at_t}

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_TRUE, ::testing::test

#include <algorithm>  // std::all_of
#include <cmath>      // std::ceil
#include <cstddef>    // std::size_t
#include <tuple>      // std::ignore
#include <vector>     // std::vector

#if __has_include("hip/hip_runtime.h") && __has_include("hip/hip_runtime_api.h")

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

template <typename T>
class HIPFillUtility : public ::testing::Test {
  protected:
    using fixture_real_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(HIPFillUtility, util::real_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(HIPFillUtility, fill_kernel) {
    using real_type = typename TestFixture::fixture_real_type;

    // allocate array on the host
    std::vector<real_type> vec(2053);

    // allocate array on the device
    real_type *vec_d{};
    std::ignore = hipMalloc((void **) &vec_d, vec.size() * sizeof(real_type));

    // create the block and grid partition
    const dim3 block{ 512 };
    const dim3 grid{ static_cast<unsigned int>(std::ceil(static_cast<double>(vec.size()) / static_cast<double>(block.x))) };

    // fill the array on the device
    plssvm::hip::detail::fill_array<<<grid, block>>>(vec_d, real_type{ 42.0 }, std::size_t{ 0 }, vec.size());
    std::ignore = hipDeviceSynchronize();

    // copy result back to the host and free device memory
    std::ignore = hipMemcpy(vec.data(), vec_d, vec.size() * sizeof(real_type), hipMemcpyDeviceToHost);
    std::ignore = hipFree(vec_d);

    // check whether all values are set correctly
    EXPECT_TRUE(std::all_of(vec.cbegin(), vec.cend(), [](const real_type val) { return val == real_type{ 42.0 }; }));
}

TYPED_TEST(HIPFillUtility, fill_kernel_partial) {
    using real_type = typename TestFixture::fixture_real_type;

    // allocate array on the host
    std::vector<real_type> vec(2053);

    // allocate array on the device
    real_type *vec_d{};
    std::ignore = hipMalloc((void **) &vec_d, vec.size() * sizeof(real_type));
    std::ignore = hipMemset(vec_d, 0, vec.size() * sizeof(real_type));

    // create the block and grid partition
    const dim3 block{ 512 };
    const dim3 grid{ static_cast<unsigned int>(std::ceil(static_cast<double>(vec.size()) / static_cast<double>(block.x))) };

    // fill the array on the device
    plssvm::hip::detail::fill_array<<<grid, block>>>(vec_d, real_type{ 3.1415 }, std::size_t{ 42 }, vec.size() - std::size_t{ 1024 });
    std::ignore = hipDeviceSynchronize();

    // copy result back to the host and free device memory
    std::ignore = hipMemcpy(vec.data(), vec_d, vec.size() * sizeof(real_type), hipMemcpyDeviceToHost);
    std::ignore = hipFree(vec_d);

    // check whether all values are set correctly
    EXPECT_TRUE(std::all_of(vec.cbegin(), vec.cbegin() + 42, [](const real_type val) { return val == real_type{ 0.0 }; }));
    EXPECT_TRUE(std::all_of(vec.cbegin() + 42, vec.cbegin() + 1071, [](const real_type val) { return val == real_type{ 3.1415 }; }));
    EXPECT_TRUE(std::all_of(vec.cbegin() + 1071, vec.cend(), [](const real_type val) { return val == real_type{ 0.0 }; }));
}

#endif