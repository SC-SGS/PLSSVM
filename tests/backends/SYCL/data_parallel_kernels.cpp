/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SYCL data parallel kernels.
 */

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::{data_parallel_kernel, list_available_sycl_data_parallel_kernels}

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT; ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_GE

#include <sstream>  // std::istringstream

// check whether the plssvm::sycl::data_parallel_kernel -> std::string conversions are correct
TEST(SYCLDataParallelKernel, ToString) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::data_parallel_kernel::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::data_parallel_kernel::basic, "basic");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::data_parallel_kernel::work_group, "work_group");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::data_parallel_kernel::hierarchical, "hierarchical");
    EXPECT_CONVERSION_TO_STRING(plssvm::sycl::data_parallel_kernel::scoped, "scoped");
}

TEST(SYCLDataParallelKernel, ToStringUnknown) {
    // check conversions to std::string from unknown file_format_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::sycl::data_parallel_kernel>(5), "unknown");
}

// check whether the std::string -> plssvm::sycl::data_parallel_kernel conversions are correct
TEST(SYCLDataParallelKernel, FromString) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::sycl::data_parallel_kernel::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOMATIC", plssvm::sycl::data_parallel_kernel::automatic);
    EXPECT_CONVERSION_FROM_STRING("auto", plssvm::sycl::data_parallel_kernel::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTO", plssvm::sycl::data_parallel_kernel::automatic);
    EXPECT_CONVERSION_FROM_STRING("basic", plssvm::sycl::data_parallel_kernel::basic);
    EXPECT_CONVERSION_FROM_STRING("BASIC", plssvm::sycl::data_parallel_kernel::basic);
    EXPECT_CONVERSION_FROM_STRING("work_group", plssvm::sycl::data_parallel_kernel::work_group);
    EXPECT_CONVERSION_FROM_STRING("WORK-GROUP", plssvm::sycl::data_parallel_kernel::work_group);
    EXPECT_CONVERSION_FROM_STRING("nd_range", plssvm::sycl::data_parallel_kernel::work_group);
    EXPECT_CONVERSION_FROM_STRING("ND-RANGE", plssvm::sycl::data_parallel_kernel::work_group);
    EXPECT_CONVERSION_FROM_STRING("hierarchical", plssvm::sycl::data_parallel_kernel::hierarchical);
    EXPECT_CONVERSION_FROM_STRING("HIERARCHICAL", plssvm::sycl::data_parallel_kernel::hierarchical);
    EXPECT_CONVERSION_FROM_STRING("scoped", plssvm::sycl::data_parallel_kernel::scoped);
    EXPECT_CONVERSION_FROM_STRING("SCOPED", plssvm::sycl::data_parallel_kernel::scoped);
}

TEST(SYCLDataParallelKernel, FromStringUnknown) {
    // foo isn't a valid file_format_type
    std::istringstream input{ "foo" };
    plssvm::sycl::data_parallel_kernel data_parallel_kernel_type{};
    input >> data_parallel_kernel_type;
    EXPECT_TRUE(input.fail());
}

TEST(SYCLDataParallelKernel, MinimalAvailableSYCLDataParallelKernels) {
    const std::vector<plssvm::sycl::data_parallel_kernel> data_parallel_kernel_types = plssvm::sycl::list_available_sycl_data_parallel_kernels();

    // at least three must be available (automatic, basic, and work_group)!
    EXPECT_GE(data_parallel_kernel_types.size(), 3);

    // check for the data parallel kernels that must always be present
    EXPECT_THAT(data_parallel_kernel_types, ::testing::Contains(plssvm::sycl::data_parallel_kernel::automatic));
    EXPECT_THAT(data_parallel_kernel_types, ::testing::Contains(plssvm::sycl::data_parallel_kernel::basic));
    EXPECT_THAT(data_parallel_kernel_types, ::testing::Contains(plssvm::sycl::data_parallel_kernel::work_group));
}
