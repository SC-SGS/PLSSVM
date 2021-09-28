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

#include "utility.hpp"  // util::gtest_expect_enum_to_string_string_conversion, util::gtest_expect_string_to_enum_conversion

#include "gtest/gtest.h"  // TEST

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
