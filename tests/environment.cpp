/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the environment setup and teardown.
 */

#include "plssvm/environment.hpp"

#include "plssvm/backend_types.hpp"          // plssvm::backend_type, plssvm::list_available_backends
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::environment_exception

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING, EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_NE, EXPECT_DEATH

#include <tuple>   // std::ignore
#include <vector>  // std::vector

// check whether the plssvm::environment::status -> std::string conversions are correct
TEST(EnvironmentStatus, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::environment::status::uninitialized, "uninitialized");
    EXPECT_CONVERSION_TO_STRING(plssvm::environment::status::initialized, "initialized");
    EXPECT_CONVERSION_TO_STRING(plssvm::environment::status::finalized, "finalized");
    EXPECT_CONVERSION_TO_STRING(plssvm::environment::status::unnecessary, "unnecessary");
}

TEST(EnvironmentStatus, to_string_unknown) {
    // check conversions to std::string from unknown environment status
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::environment::status>(4), "unknown");
}

// check whether the std::string -> plssvm::environment::status conversions are correct
TEST(EnvironmentStatus, from_string) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("uninitialized", plssvm::environment::status::uninitialized);
    EXPECT_CONVERSION_FROM_STRING("UNINITIALIZED", plssvm::environment::status::uninitialized);
    EXPECT_CONVERSION_FROM_STRING("initialized", plssvm::environment::status::initialized);
    EXPECT_CONVERSION_FROM_STRING("INITIALIZED", plssvm::environment::status::initialized);
    EXPECT_CONVERSION_FROM_STRING("finalized", plssvm::environment::status::finalized);
    EXPECT_CONVERSION_FROM_STRING("FINALIZED", plssvm::environment::status::finalized);
    EXPECT_CONVERSION_FROM_STRING("unnecessary", plssvm::environment::status::unnecessary);
    EXPECT_CONVERSION_FROM_STRING("UNNECESSARY", plssvm::environment::status::unnecessary);
}

TEST(EnvironmentStatus, from_string_unknown) {
    // foo isn't a valid environment status
    std::istringstream input{ "foo" };
    plssvm::environment::status status{};
    input >> status;
    EXPECT_TRUE(input.fail());
}

TEST(Environment, get_backend_status) {
    // check the backend statis for all supported backends

    // the automatic backend may not be used and throws an exception
    EXPECT_THROW_WHAT(std::ignore = plssvm::environment::get_backend_status(plssvm::backend_type::automatic), plssvm::environment_exception, "Can't retrieve the environment status for the automatic backend!");

    // must be always status::unnecessary for the following backends
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::openmp), plssvm::environment::status::unnecessary);
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::stdpar), plssvm::environment::status::unnecessary);
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::cuda), plssvm::environment::status::unnecessary);
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::hip), plssvm::environment::status::unnecessary);
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::opencl), plssvm::environment::status::unnecessary);
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::sycl), plssvm::environment::status::unnecessary);

    // HPX and Kokkos need some form of initialization IF THEY ARE ENABLED
#if defined(PLSSVM_HAS_HPX_BACKEND)
    EXPECT_NE(plssvm::environment::get_backend_status(plssvm::backend_type::hpx), plssvm::environment::status::unnecessary);
#else
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::hpx), plssvm::environment::status::unnecessary);
#endif

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    EXPECT_NE(plssvm::environment::get_backend_status(plssvm::backend_type::kokkos), plssvm::environment::status::unnecessary);
#else
    EXPECT_EQ(plssvm::environment::get_backend_status(plssvm::backend_type::kokkos), plssvm::environment::status::unnecessary);
#endif
}

TEST(EnvironmentDeathTest, initialize_backend) {
    // the function may never be called with the automatic backend
    EXPECT_DEATH(plssvm::environment::detail::initialize_backend(plssvm::backend_type::automatic), "The automatic backend may never be initialized!");
}

TEST(EnvironmentDeathTest, finalize_backend) {
    // the function may never be called with the automatic backend
    EXPECT_DEATH(plssvm::environment::detail::finalize_backend(plssvm::backend_type::automatic), "The automatic backend may never be finalized!");
}

TEST(Environment, initialize_impl) {
    // the function may never be called with a backend that hasn't been enabled
    const std::vector<plssvm::backend_type> all_backends{
        plssvm::backend_type::openmp,
        plssvm::backend_type::stdpar,
        plssvm::backend_type::hpx,
        plssvm::backend_type::cuda,
        plssvm::backend_type::hip,
        plssvm::backend_type::opencl,
        plssvm::backend_type::sycl,
        plssvm::backend_type::kokkos
    };
    const std::vector<plssvm::backend_type> available_backends = plssvm::list_available_backends();

    // iterate over all backends and check whether it is available
    for (const plssvm::backend_type backend : all_backends) {
        if (!plssvm::detail::contains(available_backends, backend)) {
            // backend is not available -> it cannot be initialized
            EXPECT_THROW_WHAT(plssvm::environment::detail::initialize_impl(std::vector<plssvm::backend_type>{ backend }),
                              plssvm::environment_exception,
                              fmt::format("The provided backend {} is currently not available and, therefore, can't be initialized! Available backends are: [{}].", backend, fmt::join(available_backends, ", ")));
        }
    }
}

TEST(Environment, initialize_impl_automatic) {
    // the function may never be called with the automatic backend
    const std::vector<plssvm::backend_type> backends{ plssvm::backend_type::automatic };
    EXPECT_THROW_WHAT(plssvm::environment::detail::initialize_impl(backends), plssvm::environment_exception, "The automatic backend cannot be initialized!");
}

TEST(Environment, finalize) {
    // the function may never be called with a backend that hasn't been enabled
    const std::vector<plssvm::backend_type> all_backends{
        plssvm::backend_type::openmp,
        plssvm::backend_type::stdpar,
        plssvm::backend_type::hpx,
        plssvm::backend_type::cuda,
        plssvm::backend_type::hip,
        plssvm::backend_type::opencl,
        plssvm::backend_type::sycl,
        plssvm::backend_type::kokkos
    };
    const std::vector<plssvm::backend_type> available_backends = plssvm::list_available_backends();

    // iterate over all backends and check whether it is available
    for (const plssvm::backend_type backend : all_backends) {
        if (!plssvm::detail::contains(available_backends, backend)) {
            // backend is not available -> it cannot be initialized
            EXPECT_THROW_WHAT(plssvm::environment::finalize(std::vector<plssvm::backend_type>{ backend }),
                              plssvm::environment_exception,
                              fmt::format("The provided backend {} is currently not available and, therefore, can't be finalized! Available backends are: [{}].", backend, fmt::join(available_backends, ", ")));
        }
    }
}

TEST(Environment, finalize_automatic) {
    // the function may never be called with the automatic backend
    const std::vector<plssvm::backend_type> backends{ plssvm::backend_type::automatic };
    EXPECT_THROW_WHAT(plssvm::environment::finalize(backends), plssvm::environment_exception, "The automatic backend cannot be finalized!");
}
