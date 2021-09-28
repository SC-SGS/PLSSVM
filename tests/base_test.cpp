/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the base functionality.
 */

#include "plssvm/backend_types.hpp"    // plssvm::backend_type
#include "plssvm/kernel_types.hpp"     // plssvm::kernel_type
#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include "backends/compare.hpp"  // compare::detail::linear_kernel, compare::detail::poly_kernel, compare::detail::radial_kernel
#include "utility.hpp"           // util::gtest_expect_enum_to_string_string_conversion, util::gtest_expect_string_to_enum_conversion

#include "gtest/gtest.h"  // TEST

#include <random>  // std::random_device, std::mt19937, std::uniform_real_distribution

// check whether the std::string <-> plssvm::backend_type conversions are correct
TEST(Base, backend_type) {
    // check conversions to std::string
    util::gtest_expect_enum_to_string_string_conversion(plssvm::backend_type::openmp, "openmp");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::backend_type::cuda, "cuda");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::backend_type::opencl, "opencl");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::backend_type::sycl, "sycl");
    util::gtest_expect_enum_to_string_string_conversion(static_cast<plssvm::backend_type>(4), "unknown");

    // check conversion from std::string
    util::gtest_expect_string_to_enum_conversion("openmp", plssvm::backend_type::openmp);
    util::gtest_expect_string_to_enum_conversion("OpenMP", plssvm::backend_type::openmp);
    util::gtest_expect_string_to_enum_conversion("cuda", plssvm::backend_type::cuda);
    util::gtest_expect_string_to_enum_conversion("CUDA", plssvm::backend_type::cuda);
    util::gtest_expect_string_to_enum_conversion("opencl", plssvm::backend_type::opencl);
    util::gtest_expect_string_to_enum_conversion("OpenCL", plssvm::backend_type::opencl);
    util::gtest_expect_string_to_enum_conversion("sycl", plssvm::backend_type::sycl);
    util::gtest_expect_string_to_enum_conversion("SYCL", plssvm::backend_type::sycl);
    util::gtest_expect_string_to_enum_conversion<plssvm::backend_type>("foo");
}

// check whether the std::string <-> plssvm::kernel_type conversions are correct
TEST(Base, kernel_type) {
    // check conversions to std::string
    util::gtest_expect_enum_to_string_string_conversion(plssvm::kernel_type::linear, "linear");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::kernel_type::polynomial, "polynomial");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::kernel_type::rbf, "rbf");
    util::gtest_expect_enum_to_string_string_conversion(static_cast<plssvm::kernel_type>(3), "unknown");

    // check conversion from std::string
    util::gtest_expect_string_to_enum_conversion("linear", plssvm::kernel_type::linear);
    util::gtest_expect_string_to_enum_conversion("LINEAR", plssvm::kernel_type::linear);
    util::gtest_expect_string_to_enum_conversion("0", plssvm::kernel_type::linear);
    util::gtest_expect_string_to_enum_conversion("polynomial", plssvm::kernel_type::polynomial);
    util::gtest_expect_string_to_enum_conversion("POLynomIAL", plssvm::kernel_type::polynomial);
    util::gtest_expect_string_to_enum_conversion("1", plssvm::kernel_type::polynomial);
    util::gtest_expect_string_to_enum_conversion("rbf", plssvm::kernel_type::rbf);
    util::gtest_expect_string_to_enum_conversion("rBf", plssvm::kernel_type::rbf);
    util::gtest_expect_string_to_enum_conversion("2", plssvm::kernel_type::rbf);
    util::gtest_expect_string_to_enum_conversion<plssvm::kernel_type>("bar");
}

// check whether the std::string <-> plssvm::target_platform conversions are correct
TEST(Base, target_platform) {
    // check conversions to std::string
    util::gtest_expect_enum_to_string_string_conversion(plssvm::target_platform::automatic, "automatic");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::target_platform::cpu, "cpu");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::target_platform::gpu_nvidia, "gpu_nvidia");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::target_platform::gpu_amd, "gpu_amd");
    util::gtest_expect_enum_to_string_string_conversion(plssvm::target_platform::gpu_intel, "gpu_intel");
    util::gtest_expect_enum_to_string_string_conversion(static_cast<plssvm::target_platform>(5), "unknown");

    // check conversion from std::string
    util::gtest_expect_string_to_enum_conversion("automatic", plssvm::target_platform::automatic);
    util::gtest_expect_string_to_enum_conversion("AUTOmatic", plssvm::target_platform::automatic);
    util::gtest_expect_string_to_enum_conversion("cpu", plssvm::target_platform::cpu);
    util::gtest_expect_string_to_enum_conversion("CPU", plssvm::target_platform::cpu);
    util::gtest_expect_string_to_enum_conversion("gpu_nvidia", plssvm::target_platform::gpu_nvidia);
    util::gtest_expect_string_to_enum_conversion("GPU_NVIDIA", plssvm::target_platform::gpu_nvidia);
    util::gtest_expect_string_to_enum_conversion("gpu_amd", plssvm::target_platform::gpu_amd);
    util::gtest_expect_string_to_enum_conversion("GPU_AMD", plssvm::target_platform::gpu_amd);
    util::gtest_expect_string_to_enum_conversion("gpu_intel", plssvm::target_platform::gpu_intel);
    util::gtest_expect_string_to_enum_conversion("GPU_INTEL", plssvm::target_platform::gpu_intel);
    util::gtest_expect_string_to_enum_conversion<plssvm::target_platform>("baz");
}

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class BaseKernelFunction : public ::testing::Test {};
TYPED_TEST_SUITE(BaseTransform, floating_point_types);

// check whether the kernel_function implementation is correct
TEST(BaseKernelFunction, kernel_function) {
    using real_type = double;

    // create dummy data vectors
    constexpr std::size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);

    // fill vectors with random values
    std::random_device rnd_device;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(1.0, 2.0);
    std::generate(x1.begin(), x1.end(), [&]() { return dist(gen); });
    std::generate(x2.begin(), x2.end(), [&]() { return dist(gen); });

    util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::linear>(x1, x2), compare::detail::linear_kernel(x1, x2));
    util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::polynomial>(x1, x2, 3, 0.5, 0.0), compare::detail::poly_kernel(x1, x2, 3, 0.5, 0.0));
    util::gtest_assert_floating_point_near(plssvm::kernel_function<plssvm::kernel_type::rbf>(x1, x2, 0.5), compare::detail::radial_kernel(x1, x2, 0.5));
}