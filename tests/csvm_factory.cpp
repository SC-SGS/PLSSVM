/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the CSVM factory function.
 */

#include "plssvm/csvm_factory.hpp"

#include "plssvm/backend_types.hpp"          // plssvm::backend_type, plssvm::csvm_to_backend_type_v
#include "plssvm/csvm.hpp"                   // plssvm::csvm_backend_exists_v
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT_MATCHER
#include "utility.hpp"             // util::redirect_output

#include "fmt/core.h"              // fmt::format
#include "fmt/ostream.h"           // be able to format a plssvm::backend_type using fmt
#include "gmock/gmock-matchers.h"  // ::testing::StartsWith
#include "gtest/gtest.h"           // TYPED_TEST_SUITE, TYPED_TEST, ::testing::{Test, Types, internal::GetTypeName}

#include <string>  // std::string
#include <tuple>   // std::ignore

// clang-format off
using csvm_types = ::testing::Types<plssvm::openmp::csvm,
                                    plssvm::cuda::csvm, plssvm::hip::csvm, plssvm::opencl::csvm,
                                    plssvm::sycl::csvm, plssvm::hipsycl::csvm, plssvm::dpcpp::csvm>;
// clang-format on

namespace testing::internal {  // dirty hack to have type names for incomplete types
template <>
std::string GetTypeName<plssvm::openmp::csvm>() { return "openmp_csvm"; }
template <>
std::string GetTypeName<plssvm::cuda::csvm>() { return "cuda_csvm"; }
template <>
std::string GetTypeName<plssvm::hip::csvm>() { return "hip_csvm"; }
template <>
std::string GetTypeName<plssvm::opencl::csvm>() { return "opencl_csvm"; }
template <>
std::string GetTypeName<plssvm::dpcpp::csvm>() { return "sycl_dpcpp_csvm"; }
template <>
std::string GetTypeName<plssvm::hipsycl::csvm>() { return "sycl_hipsycl_csvm"; }
}  // namespace testing::internal

template <typename T>
class CSVMFactory : public ::testing::Test, private util::redirect_output {};
TYPED_TEST_SUITE(CSVMFactory, csvm_types);

TYPED_TEST(CSVMFactory, factory_backend) {
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_default) {
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm());
}
TYPED_TEST(CSVMFactory, factory_backend_parameter) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, params);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, params),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_parameter) {
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(params));
}
TYPED_TEST(CSVMFactory, factory_backend_target) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, target),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_target) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(target));
}
TYPED_TEST(CSVMFactory, factory_backend_target_and_parameter) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, params);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, target, params),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_target_and_parameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(target, params));
}

TYPED_TEST(CSVMFactory, factory_backend_target_and_named_parameter) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_target_and_named_parameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}
TYPED_TEST(CSVMFactory, factory_backend_named_parameter) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<TypeParam>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}
TEST(CSVMFactory, factory_named_parameter) {
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

using sycl_csvm_types = ::testing::Types<plssvm::sycl::csvm, plssvm::hipsycl::csvm, plssvm::dpcpp::csvm>;

template <typename T>
class SYCLCSVMFactory : public CSVMFactory<T> {};
TYPED_TEST_SUITE(SYCLCSVMFactory, sycl_csvm_types);

TYPED_TEST(SYCLCSVMFactory, factory_sycl_implementation) {
    // the backend to use
    const plssvm::backend_type backend = plssvm::backend_type::sycl;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<TypeParam>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<TypeParam>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(TypeParam, csvm);
    } else {
        EXPECT_THROW_WHAT_MATCHER(std::ignore = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<TypeParam>::impl),
                                  plssvm::unsupported_backend_exception,
                                  ::testing::StartsWith(fmt::format("No {} backend available", backend)));
    }
}