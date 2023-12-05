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
#include "types_to_test.hpp"       // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, test_parameter_type_at_t}

#include "fmt/core.h"              // fmt::format
#include "fmt/ostream.h"           // be able to format a plssvm::backend_type using fmt
#include "gmock/gmock-matchers.h"  // ::testing::StartsWith
#include "gtest/gtest.h"           // TYPED_TEST_SUITE, TYPED_TEST, ::testing::{Test, Types, internal::GetTypeName}

#include <string>  // std::string
#include <tuple>   // std::tuple, std::ignore

namespace util {

// clang-format off
using csvm_types = std::tuple<plssvm::openmp::csvm, plssvm::cuda::csvm, plssvm::hip::csvm, plssvm::opencl::csvm, plssvm::sycl::csvm>;
using csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<csvm_types>>;
/// A type list of all supported SYCL C-SVMs.
using sycl_csvm_types = std::tuple<plssvm::sycl::csvm, plssvm::adaptivecpp::csvm, plssvm::dpcpp::csvm>;
using sycl_csvm_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<sycl_csvm_types>>;
// clang-format on

}  // namespace util

namespace testing::internal {  // dirty hack to have type names for incomplete types
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::openmp::csvm>, util::value_list<>>>() { return "openmp_csvm"; }
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::cuda::csvm>, util::value_list<>>>() { return "cuda_csvm"; }
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::hip::csvm>, util::value_list<>>>() { return "hip_csvm"; }
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::opencl::csvm>, util::value_list<>>>() { return "opencl_csvm"; }
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::dpcpp::csvm>, util::value_list<>>>() { return "sycl_dpcpp_csvm"; }
template <>
std::string GetTypeName<util::test_parameter<util::type_list<plssvm::adaptivecpp::csvm>, util::value_list<>>>() { return "sycl_adaptivecpp_csvm"; }
}  // namespace testing::internal

template <typename T>
class CSVMFactory : public ::testing::Test, private util::redirect_output<> {
  protected:
    using fixture_backend_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(CSVMFactory, util::csvm_types_gtest);

TYPED_TEST(CSVMFactory, factory_backend) {
    using backend_type = typename TestFixture::fixture_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TEST(CSVMFactory, factory_default) {
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm());
}
TYPED_TEST(CSVMFactory, factory_backend_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, params);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TEST(CSVMFactory, factory_parameter) {
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(params));
}
TYPED_TEST(CSVMFactory, factory_backend_target) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TEST(CSVMFactory, factory_target) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(target));
}
TYPED_TEST(CSVMFactory, factory_backend_target_and_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, params);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
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
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
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
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TEST(CSVMFactory, factory_named_parameter) {
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVM creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

TEST(CSVMFactory, invalid_backend) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(static_cast<plssvm::backend_type>(6)),
                      plssvm::unsupported_backend_exception,
                      "Unrecognized backend provided!");
}
TEST(CSVMFactory, invalid_constructor_parameter) {
    if constexpr (plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>) {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(plssvm::backend_type::cuda, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::automatic),
                          plssvm::unsupported_backend_exception,
                          "Provided invalid (named) arguments for the cuda backend!");
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(plssvm::backend_type::cuda),
                          plssvm::unsupported_backend_exception,
                          "No cuda backend available!");
    }
}

template <typename T>
class SYCLCSVMFactory : public CSVMFactory<T> {
  protected:
    using fixture_sycl_backend_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(SYCLCSVMFactory, util::sycl_csvm_types_gtest);

TYPED_TEST(SYCLCSVMFactory, factory_backend) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TYPED_TEST(SYCLCSVMFactory, factory_backend_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TYPED_TEST(SYCLCSVMFactory, factory_backend_target) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TYPED_TEST(SYCLCSVMFactory, factory_backend_target_and_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVMFactory, factory_backend_target_and_named_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}
TYPED_TEST(SYCLCSVMFactory, factory_backend_named_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm
        const auto csvm = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        // check whether the created csvm has the same type as the expected one
        EXPECT_INSTANCE_OF(sycl_backend_type, csvm);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(SYCLCSVMFactory, invalid_sycl_implementation) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = static_cast<plssvm::sycl::implementation_type>(3)),
                      plssvm::unsupported_backend_exception,
                      "No sycl backend available!");
}