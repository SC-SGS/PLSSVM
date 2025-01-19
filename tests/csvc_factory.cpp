/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the CSVM factory function creating a C-SVC.
 */

#include "plssvm/backend_types.hpp"                       // plssvm::backend_type, plssvm::csvm_to_backend_type_v
#include "plssvm/backends/SYCL/implementation_types.hpp"  // plssvm::sycl::implementation_type
#include "plssvm/csvm_factory.hpp"                        // factory functions to test
#include "plssvm/exceptions/exceptions.hpp"               // plssvm::unsupported_backend_exception
#include "plssvm/kernel_function_types.hpp"               // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                           // plssvm::parameter
#include "plssvm/svm/csvc.hpp"                            // plssvm::csvc
#include "plssvm/svm/csvm.hpp"                            // plssvm::csvm_backend_exists_v
#include "plssvm/target_platforms.hpp"                    // plssvm::target_platform

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT_MATCHER
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{combine_test_parameters_gtest_t, cartesian_type_product_t, test_parameter_type_at_t}
#include "tests/utility.hpp"             // util::redirect_output

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST_SUITE, TYPED_TEST, ::testing::{Test, Types, internal::GetTypeName}

#include <tuple>  // std::tuple, std::ignore

namespace util {

/// A type list of all supported C-SVRs.
using csvc_types = std::tuple<plssvm::openmp::csvc, plssvm::hpx::csvc, plssvm::stdpar::csvc, plssvm::cuda::csvc, plssvm::hip::csvc, plssvm::opencl::csvc, plssvm::sycl::csvc, plssvm::kokkos::csvc>;
using csvc_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<csvc_types>>;

/// A type list of all supported SYCL C-SVRs.
using sycl_csvc_types = std::tuple<plssvm::sycl::csvc, plssvm::adaptivecpp::csvc, plssvm::dpcpp::csvc>;
using sycl_csvc_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<sycl_csvc_types>>;

}  // namespace util

template <typename T>
class CSVCFactory : public ::testing::Test,
                    private util::redirect_output<> {
  protected:
    using fixture_backend_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(CSVCFactory, util::csvc_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(CSVCFactory, factory_backend) {
    using backend_type = typename TestFixture::fixture_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_default) {
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc());
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>());
}

TYPED_TEST(CSVCFactory, factory_backend_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, params);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, params);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_parameter) {
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc(params));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>(params));
}

TYPED_TEST(CSVCFactory, factory_backend_target) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_target) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc(target));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>(target));
}

TYPED_TEST(CSVCFactory, factory_backend_target_and_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target, params);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target, params);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_target_and_parameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc(target, params));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>(target, params));
}

TYPED_TEST(CSVCFactory, factory_backend_target_and_named_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_target_and_named_parameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc(target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>(target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

TYPED_TEST(CSVCFactory, factory_backend_named_parameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVCFactory, factory_named_parameter) {
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvc(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvc>(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

TEST(CSVCFactory, invalid_backend) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(static_cast<plssvm::backend_type>(9)),
                      plssvm::unsupported_backend_exception,
                      "Unrecognized backend provided!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(static_cast<plssvm::backend_type>(9)),
                      plssvm::unsupported_backend_exception,
                      "Unrecognized backend provided!");
}

TEST(CSVCFactory, unsupported_backend) {
    if constexpr (plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>) {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(plssvm::backend_type::cuda, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::automatic),
                          plssvm::unsupported_backend_exception,
                          "Provided invalid (named) arguments for the cuda backend!");
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(plssvm::backend_type::cuda, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::automatic),
                          plssvm::unsupported_backend_exception,
                          "Provided invalid (named) arguments for the cuda backend!");
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(plssvm::backend_type::cuda),
                          plssvm::unsupported_backend_exception,
                          "No cuda backend available!");
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(plssvm::backend_type::cuda),
                          plssvm::unsupported_backend_exception,
                          "No cuda backend available!");
    }
}

template <typename T>
class SYCLCSVCFactory : public CSVCFactory<T> {
  protected:
    using fixture_sycl_backend_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(SYCLCSVCFactory, util::sycl_csvc_types_gtest);

TYPED_TEST(SYCLCSVCFactory, factory_backend) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVCFactory, factory_backend_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVCFactory, factory_backend_target) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVCFactory, factory_backend_target_and_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVCFactory, factory_backend_target_and_named_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVCFactory, factory_backend_named_parameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvc1 = plssvm::make_csvc(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc1);
        const auto csvc2 = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvc2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(SYCLCSVCFactory, invalid_sycl_implementation) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvc(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = static_cast<plssvm::sycl::implementation_type>(3)),
                      plssvm::unsupported_backend_exception,
                      "No sycl backend available!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvc>(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = static_cast<plssvm::sycl::implementation_type>(3)),
                      plssvm::unsupported_backend_exception,
                      "No sycl backend available!");
}
