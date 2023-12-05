/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different backend_types.
 */

#include "plssvm/backend_types.hpp"

#include "plssvm/backends/SYCL/detail/constants.hpp"  // namespace plssvm::sycl
#include "plssvm/detail/utility.hpp"                  // plssvm::detail::contains

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::{pretty_print_unsupported_backend_combination, pretty_print_supported_backend_combination}

#include "fmt/core.h"     // fmt::format
#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_GE, TEST_P, INSTANTIATE_TEST_SUITE_P

#include <sstream>  // std::istringstream
#include <tuple>    // std::tuple, std::ignore
#include <utility>  // std::pair
#include <vector>   // std::vector

// check whether the plssvm::backend_type -> std::string conversions are correct
TEST(BackendType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::automatic, "automatic");
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::openmp, "openmp");
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::cuda, "cuda");
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::hip, "hip");
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::opencl, "opencl");
    EXPECT_CONVERSION_TO_STRING(plssvm::backend_type::sycl, "sycl");
}
TEST(BackendType, to_string_unknown) {
    // check conversions to std::string from unknown backend_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::backend_type>(6), "unknown");
}

// check whether the std::string -> plssvm::backend_type conversions are correct
TEST(BackendType, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("automatic", plssvm::backend_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("AUTOmatic", plssvm::backend_type::automatic);
    EXPECT_CONVERSION_FROM_STRING("openmp", plssvm::backend_type::openmp);
    EXPECT_CONVERSION_FROM_STRING("OpenMP", plssvm::backend_type::openmp);
    EXPECT_CONVERSION_FROM_STRING("cuda", plssvm::backend_type::cuda);
    EXPECT_CONVERSION_FROM_STRING("CUDA", plssvm::backend_type::cuda);
    EXPECT_CONVERSION_FROM_STRING("hip", plssvm::backend_type::hip);
    EXPECT_CONVERSION_FROM_STRING("HIP", plssvm::backend_type::hip);
    EXPECT_CONVERSION_FROM_STRING("opencl", plssvm::backend_type::opencl);
    EXPECT_CONVERSION_FROM_STRING("OpenCL", plssvm::backend_type::opencl);
    EXPECT_CONVERSION_FROM_STRING("sycl", plssvm::backend_type::sycl);
    EXPECT_CONVERSION_FROM_STRING("SYCL", plssvm::backend_type::sycl);
}
TEST(BackendType, from_string_unknown) {
    // foo isn't a valid backend_type
    std::istringstream input{ "foo" };
    plssvm::backend_type backend{};
    input >> backend;
    EXPECT_TRUE(input.fail());
}

TEST(BackendType, minimal_available_backend) {
    const std::vector<plssvm::backend_type> backends = plssvm::list_available_backends();

    // at least two backends must be available (automatic + one user provided)!
    EXPECT_GE(backends.size(), 2);

    // the automatic backend must always be present
    EXPECT_THAT(backends, ::testing::Contains(plssvm::backend_type::automatic));
}

TEST(BackendType, determine_default_backend_type) {
    // the determined default backend must not be backend_type::automatic
    const plssvm::backend_type backend = plssvm::determine_default_backend();
    EXPECT_NE(backend, plssvm::backend_type::automatic);
}

using unsupported_combination_type = std::pair<std::vector<plssvm::backend_type>, std::vector<plssvm::target_platform>>;
class BackendTypeUnsupportedCombination : public ::testing::TestWithParam<unsupported_combination_type> {};

TEST_P(BackendTypeUnsupportedCombination, unsupported_backend_target_platform_combinations) {
    const auto &[available_backends, available_target_platforms] = GetParam();
    EXPECT_THROW_WHAT(std::ignore = plssvm::determine_default_backend(available_backends, available_target_platforms),
                      plssvm::unsupported_backend_exception,
                      fmt::format("Error: unsupported backend and target platform combination: [{}]x[{}]!", fmt::join(available_backends, ", "), fmt::join(available_target_platforms, ", ")));
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(BackendType, BackendTypeUnsupportedCombination, ::testing::Values(
         unsupported_combination_type{ { plssvm::backend_type::cuda, plssvm::backend_type::hip }, { plssvm::target_platform::cpu } },
         unsupported_combination_type{ { plssvm::backend_type::openmp }, { plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel } },
         unsupported_combination_type{ { plssvm::backend_type::cuda }, { plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel } },
         unsupported_combination_type{ { plssvm::backend_type::hip }, { plssvm::target_platform::gpu_intel } }),
         naming::pretty_print_unsupported_backend_combination<BackendTypeUnsupportedCombination>);
// clang-format on

using supported_combination_type = std::tuple<std::vector<plssvm::backend_type>, std::vector<plssvm::target_platform>, plssvm::backend_type>;
class BackendTypeSupportedCombination : public ::testing::TestWithParam<supported_combination_type> {};

TEST_P(BackendTypeSupportedCombination, supported_backend_target_platform_combinations) {
    const auto &[available_backends, available_target_platforms, result_backend] = GetParam();
    EXPECT_EQ(plssvm::determine_default_backend(available_backends, available_target_platforms), result_backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(BackendType, BackendTypeSupportedCombination, ::testing::Values(
         supported_combination_type{ { plssvm::backend_type::openmp }, { plssvm::target_platform::cpu, plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel }, plssvm::backend_type::openmp },
         supported_combination_type{ { plssvm::backend_type::cuda }, { plssvm::target_platform::cpu, plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel }, plssvm::backend_type::cuda },
         supported_combination_type{ { plssvm::backend_type::hip }, { plssvm::target_platform::cpu, plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel }, plssvm::backend_type::hip },
         supported_combination_type{ { plssvm::backend_type::opencl }, { plssvm::target_platform::cpu, plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel }, plssvm::backend_type::opencl },
         supported_combination_type{ { plssvm::backend_type::sycl }, { plssvm::target_platform::cpu, plssvm::target_platform::gpu_nvidia, plssvm::target_platform::gpu_amd, plssvm::target_platform::gpu_intel }, plssvm::backend_type::sycl },
         supported_combination_type{ { plssvm::backend_type::openmp, plssvm::backend_type::cuda, plssvm::backend_type::hip, plssvm::backend_type::opencl, plssvm::backend_type::sycl }, { plssvm::target_platform::cpu }, plssvm::backend_type::sycl },
         supported_combination_type{ { plssvm::backend_type::openmp, plssvm::backend_type::cuda, plssvm::backend_type::hip, plssvm::backend_type::opencl, plssvm::backend_type::sycl }, { plssvm::target_platform::gpu_nvidia }, plssvm::backend_type::cuda },
         supported_combination_type{ { plssvm::backend_type::openmp, plssvm::backend_type::cuda, plssvm::backend_type::hip, plssvm::backend_type::opencl, plssvm::backend_type::sycl }, { plssvm::target_platform::gpu_amd }, plssvm::backend_type::hip },
         supported_combination_type{ { plssvm::backend_type::openmp, plssvm::backend_type::cuda, plssvm::backend_type::hip, plssvm::backend_type::opencl, plssvm::backend_type::sycl }, { plssvm::target_platform::gpu_intel }, plssvm::backend_type::sycl }),
         naming::pretty_print_supported_backend_combination<BackendTypeSupportedCombination>);
// clang-format on

TEST(BackendType, csvm_to_backend_type) {
    // test the type_trait
    EXPECT_EQ(plssvm::csvm_to_backend_type<plssvm::openmp::csvm>::value, plssvm::backend_type::openmp);
    EXPECT_EQ(plssvm::csvm_to_backend_type<const plssvm::cuda::csvm>::value, plssvm::backend_type::cuda);
    EXPECT_EQ(plssvm::csvm_to_backend_type<plssvm::hip::csvm &>::value, plssvm::backend_type::hip);
    EXPECT_EQ(plssvm::csvm_to_backend_type<const plssvm::opencl::csvm &>::value, plssvm::backend_type::opencl);
    EXPECT_EQ(plssvm::csvm_to_backend_type<volatile plssvm::sycl::csvm>::value, plssvm::backend_type::sycl);
    EXPECT_EQ(plssvm::csvm_to_backend_type<const volatile plssvm::adaptivecpp::csvm>::value, plssvm::backend_type::sycl);
    EXPECT_EQ(plssvm::csvm_to_backend_type<const volatile plssvm::dpcpp::csvm &>::value, plssvm::backend_type::sycl);

    EXPECT_EQ(plssvm::csvm_to_backend_type<plssvm::adaptivecpp::csvm>::impl, plssvm::sycl::implementation_type::adaptivecpp);
    EXPECT_EQ(plssvm::csvm_to_backend_type<plssvm::dpcpp::csvm>::impl, plssvm::sycl::implementation_type::dpcpp);
}
TEST(BackendType, csvm_to_backend_type_v) {
    // test the type_trait
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<plssvm::openmp::csvm>, plssvm::backend_type::openmp);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<const plssvm::cuda::csvm>, plssvm::backend_type::cuda);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<plssvm::hip::csvm &>, plssvm::backend_type::hip);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<const plssvm::opencl::csvm &>, plssvm::backend_type::opencl);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<volatile plssvm::sycl::csvm>, plssvm::backend_type::sycl);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<const volatile plssvm::adaptivecpp::csvm>, plssvm::backend_type::sycl);
    EXPECT_EQ(plssvm::csvm_to_backend_type_v<const volatile plssvm::dpcpp::csvm &>, plssvm::backend_type::sycl);
}