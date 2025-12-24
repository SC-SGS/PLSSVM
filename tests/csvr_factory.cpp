/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the C-SVM factory function creating a C-SVR.
 */

#include "plssvm/backend_types.hpp"                       // plssvm::backend_type, plssvm::csvm_to_backend_type_v
#include "plssvm/backends/SYCL/implementation_types.hpp"  // plssvm::sycl::implementation_type
#include "plssvm/csvm_factory.hpp"                        // factory functions to test
#include "plssvm/exceptions/exceptions.hpp"               // plssvm::unsupported_backend_exception
#include "plssvm/kernel_function_types.hpp"               // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                           // plssvm::parameter
#include "plssvm/svm/csvm.hpp"                            // plssvm::csvm_backend_exists_v
#include "plssvm/svm/csvr.hpp"                            // plssvm::csvr
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
using csvr_types = std::tuple<plssvm::openmp::csvr, plssvm::hpx::csvr, plssvm::stdpar::csvr, plssvm::cuda::csvr, plssvm::hip::csvr, plssvm::opencl::csvr, plssvm::sycl::csvr, plssvm::kokkos::csvr>;
using csvr_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<csvr_types>>;

/// A type list of all supported SYCL C-SVRs.
using sycl_csvr_types = std::tuple<plssvm::sycl::csvr, plssvm::adaptivecpp::csvr, plssvm::dpcpp::csvr>;
using sycl_csvr_types_gtest = util::combine_test_parameters_gtest_t<util::cartesian_type_product_t<sycl_csvr_types>>;

}  // namespace util

template <typename T>
class CSVRFactory : public ::testing::Test,
                    private util::redirect_output<> {
  protected:
    using fixture_backend_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(CSVRFactory, util::csvr_types_gtest, naming::test_parameter_to_name);

TYPED_TEST(CSVRFactory, FactoryBackend) {
    using backend_type = typename TestFixture::fixture_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryDefault) {
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr());
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>());
}

TYPED_TEST(CSVRFactory, FactoryBackendParameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, params);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, params);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryParameter) {
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr(params));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>(params));
}

TYPED_TEST(CSVRFactory, FactoryBackendTarget) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryTarget) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr(target));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>(target));
}

TYPED_TEST(CSVRFactory, FactoryBackendTargetAndParameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target, params);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target, params);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target, params),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryTargetAndParameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr(target, params));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>(target, params));
}

TYPED_TEST(CSVRFactory, FactoryBackendTargetAndNamedParameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryTargetAndNamedParameter) {
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr(target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>(target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

TYPED_TEST(CSVRFactory, FactoryBackendNamedParameter) {
    using backend_type = typename TestFixture::fixture_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01);
        EXPECT_INSTANCE_OF(backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(CSVRFactory, FactoryNamedParameter) {
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    // with the automatic backend type there MUST be a C-SVR creatable
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvr(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
    EXPECT_NO_THROW(std::ignore = plssvm::make_csvm<plssvm::csvr>(plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01));
}

TEST(CSVRFactory, InvalidBackend) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(static_cast<plssvm::backend_type>(9)),
                      plssvm::unsupported_backend_exception,
                      "Unrecognized backend provided!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(static_cast<plssvm::backend_type>(9)),
                      plssvm::unsupported_backend_exception,
                      "Unrecognized backend provided!");
}

TEST(CSVRFactory, UnsupportedBackend) {
    if constexpr (plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>) {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(plssvm::backend_type::cuda, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::automatic),
                          plssvm::unsupported_backend_exception,
                          "Provided invalid (named) arguments for the cuda backend!");
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(plssvm::backend_type::cuda, plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::automatic),
                          plssvm::unsupported_backend_exception,
                          "Provided invalid (named) arguments for the cuda backend!");
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(plssvm::backend_type::cuda),
                          plssvm::unsupported_backend_exception,
                          "No cuda backend available!");
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(plssvm::backend_type::cuda),
                          plssvm::unsupported_backend_exception,
                          "No cuda backend available!");
    }
}

template <typename T>
class SYCLCSVRFactory : public CSVRFactory<T> {
  protected:
    using fixture_sycl_backend_type = util::test_parameter_type_at_t<0, T>;
};

TYPED_TEST_SUITE(SYCLCSVRFactory, util::sycl_csvr_types_gtest);

TYPED_TEST(SYCLCSVRFactory, FactoryBackend) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVRFactory, FactoryBackendParameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVRFactory, FactoryBackendTarget) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVRFactory, FactoryBackendTargetAndParameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // create the parameter class used
    const plssvm::parameter params{};
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target, params, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVRFactory, FactoryBackendTargetAndNamedParameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the target platform to use
    const plssvm::target_platform target = plssvm::target_platform::automatic;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, target, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TYPED_TEST(SYCLCSVRFactory, FactoryBackendNamedParameter) {
    using sycl_backend_type = typename TestFixture::fixture_sycl_backend_type;

    // the backend to use
    const plssvm::backend_type backend = plssvm::csvm_to_backend_type_v<sycl_backend_type>;
    // the kernel function to use
    const plssvm::kernel_function_type kernel_type = plssvm::kernel_function_type::polynomial;
    if constexpr (plssvm::csvm_backend_exists_v<sycl_backend_type>) {
        // create csvm and check whether the created csvm has the same type as the expected one
        const auto csvr1 = plssvm::make_csvr(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr1);
        const auto csvr2 = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl);
        EXPECT_INSTANCE_OF(sycl_backend_type, csvr2);
    } else {
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
        EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(backend, plssvm::kernel_type = kernel_type, plssvm::gamma = 0.01, plssvm::sycl_implementation_type = plssvm::csvm_to_backend_type<sycl_backend_type>::impl),
                          plssvm::unsupported_backend_exception,
                          fmt::format("No {} backend available!", backend));
    }
}

TEST(SYCLCSVRFactory, InvalidSYCLImplementation) {
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvr(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = static_cast<plssvm::sycl::implementation_type>(3)),
                      plssvm::unsupported_backend_exception,
                      "No sycl backend available!");
    EXPECT_THROW_WHAT(std::ignore = plssvm::make_csvm<plssvm::csvr>(plssvm::backend_type::sycl, plssvm::sycl_implementation_type = static_cast<plssvm::sycl::implementation_type>(3)),
                      plssvm::unsupported_backend_exception,
                      "No sycl backend available!");
}
